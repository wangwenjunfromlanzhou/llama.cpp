# Gemma 推理流程：开启 DSpark vs 不开启 DSpark（小白入门版）

> 适用代码版本：`D:\code\llama.cpp-wwj2`，分支 `master`
> 涉及代码：
> - 推理图：`src/models/gemma4.cpp`、`src/models/dspark.cpp`、`src/models/dflash.cpp`
> - 投机驱动：`common/speculative.cpp`
> - 转换器：`conversion/gemma.py`（`Gemma4DSparkModel`）
> - 调用层：`examples/speculative-simple/speculative-simple.cpp`、`tools/server/server-context.cpp`

本文档面向**第一次接触投机解码的新手**，假设你：
- 会读 C++/Python，但没读过 llama.cpp 内部
- 知道 LLM 是"输入一段话，输出一段话"，但不清楚内部 token、KV cache、采样这些细节
- 想搞懂 DSpark 在 Gemma 上是怎么工作的，以及为什么"接受率"会不够

阅读本文后，你应当能：
1. 说清楚"开 DSpark"和"不开 DSpark"两种推理路径有什么不同
2. 解释什么是接受率、它受哪些因素影响
3. 在接受率异常时知道从哪些代码位置入手排查

---

## 第 0 章：先认识几个基础概念

在讲 DSpark 之前，先把后面要用的术语过一遍。**如果你已经懂，可以直接跳到第 1 章。**

### 0.1 token、vocab、token id

大模型不直接处理"字"，而是处理"token"。token 可以是一个字、半个字、一个词或一个标点。

- **vocab（词表）**：模型认识的所有 token 的集合。Gemma4 词表大小约 262400。
- **token id**：每个 token 在词表里的编号，从 0 开始。
- **`tok_embd`**：大小为 `[n_vocab, n_embd]` 的矩阵。给定 token id `i`，查表得到它的"向量表示"（embedding），就是 `tok_embd` 的第 `i` 行。

> 类比：vocab 是一本字典，token id 是页码，`tok_embd` 是字典里每一页上写的"含义向量"。

### 0.2 logits、softmax、argmax、采样

模型最后一层会为"下一个 token 可能是什么"给每个 token 打一个分数，这个分数向量叫 **logits**，长度等于 `n_vocab`。

- **softmax**：把 logits 转成概率（每个分数变成 0~1 之间，所有分数加起来等于 1）。
- **argmax**：选出分数最高的那个 token。这是"贪心解码"（greedy）。
- **采样（sampling）**：按概率分布随机抽一个 token。这是"采样解码"。常用策略：
  - **temperature**：>1 让分布更平均（更有创造力），<1 让分布更尖锐（更确定），=0 等价于 argmax。
  - **top_k**：只在分数最高的 k 个里挑。
  - **top_p**：只在累计概率达到 p 的那几个最高分 token 里挑。

**关键认识**：采样器选的 token 不一定是 argmax 选的 token。这在 DSpark 里是个大坑（见第 4 章）。

### 0.3 autoregressive 解码

LLM 一次只生成一个 token。流程：

```
[已有: 今天] → 模型 → 预测下一个 → "天气"
[已有: 今天天气] → 模型 → 预测下一个 → "真"
[已有: 今天天气真] → 模型 → 预测下一个 → "好"
...
```

每一步都把"目前所有 token"喂给模型，得到下一个 token，把它接在末尾，再喂回去。这叫 **自回归（autoregressive）**。

### 0.4 KV cache（关键缓存）

模型在每一步都要看前面所有的 token。如果每步都重算所有历史 token 的 attention，那 O(N²) 计算量会爆炸。

**KV cache** 就是把每一层对历史 token 算出来的 K（Key）和 V（Value）向量缓存起来，下一步直接拿来用，不用重算。

- 输入 `n_tokens` 个 token → 每层缓存 `n_tokens` 行 K 和 V
- 下一步生成时，新 token 只需要算自己的 Q（Query），和缓存里所有 K 算 attention，得到加权 V

**关键认识**：KV cache 是"历史信息"。如果缓存里某些位置是空的或错的，模型看到的上下文就缺一块或错位，输出就会乱。

### 0.5 RMSNorm（Gemma 特有）

每一层前后都要做归一化（normalization）。Gemma 用的是 **RMSNorm**，公式（HF 实现里）是：

```
output = x * (1 + weight) / sqrt(mean(x²) + eps)
```

注意那个 **`(1 + weight)`**！这是 Gemma 家族的"祖传"做法。

**为什么要注意**：因为代码里 `weight` 实际存的是 `weight - 1` 的版本（HF 里 `1+w`），所以在把权重从 HF 转到 GGUF 时，必须做 **`weight + 1` 偏移**（见 `conversion/gemma.py:125`）。如果偏移没做或做错，模型输出会偏一倍。

### 0.6 RoPE（旋转位置编码）

注意力机制本身不知道 token 的顺序。为了让模型知道"先有'今天'，再有'天气'"，要在每个 token 的 Q、K 向量上叠加一个"位置信号"。

Gemma 用 **RoPE**：把 Q、K 向量按 token 位置旋转一个角度，位置越靠后旋转越多。这样两个 token 算点积时，"距离近的"会有更高的相似度。

**关键认识**：每个 token 在 K/Q 上用哪个 RoPE 角度，取决于它**在序列中的位置**（`pos`）。如果位置算错，attention 全乱。

### 0.7 causal attention（因果注意力）

训练时不能让"今天的 token"看到"明天的 token"（否则就是作弊）。所以 attention 上有一个**三角掩码**：

```
位置:    0 1 2 3
token 0: ✓ . . .    ← 只能看自己
token 1: ✓ ✓ . .    ← 只能看 0、1
token 2: ✓ ✓ ✓ .
token 3: ✓ ✓ ✓ ✓
```

这叫 **causal mask**。

**关键认识**：训练时 causal，但 **DSpark 的 draft 模型在校验时必须关闭 causal**（见第 4 章 4.6）。因为 draft 一次要预测整个 block，每个位置都要看到整个 block 的上下文（包括"未来"的 mask）。

---

## 第 1 章：不开 DSpark 的推理流程（基线）

这是标准 autoregressive，每生成 1 个 token 跑 1 次完整 forward：

```
for each step:
    1. 准备 batch:  [id_last]                  (只有 1 个 token)
    2. llama_decode(ctx_tgt)
       ├─ build_inp_embd:      tok_embd[id_last]      ← 查表得 embedding
       ├─ for il in n_layer:                            ← Gemma4 有几十层
       │     attn_norm → wq/wk/wv → q_norm/k_norm → rope → attn → wo
       │     ffn_norm → ffn(gate, up, down) → residual
       │     (Gemma4: 可能还有 attn_post_norm / ffn_post_norm / out_scale)
       └─ output_norm → lm_head → logits[n_vocab]
    3. sampler.sample(logits) → id_next          ← 选出下一个 token
    4. id_last = id_next; n_past++
```

特点：
- **简单可靠**：没有投机，没有 batch 校验，没有回滚。
- **慢**：每个 token 都要跑一遍几十层网络。
- **KV cache 单调追加**：每步只往末尾加一行，不需要删除。

性能瓶颈：**算力受限于目标模型 forward 一次的耗时**。

---

## 第 2 章：投机解码（speculative decoding）的核心思想

**问题**：能不能让一个"小模型"先猜几个 token，然后让"大模型"一次性校验？

**关键洞察**：大模型一次校验 N 个 token 的成本，**远小于** 它单独跑 N 次的成本。因为：
- 大模型 forward 一次的开销主要在"矩阵乘法"上
- 矩阵乘法的耗时跟 batch size 不严格成正比（GPU 有并行能力）
- 所以 batch=10 的 forward 可能只比 batch=1 慢 2~3 倍，而不是 10 倍

如果小模型猜得准，**大模型用 1 次 forward 就能确认多个 token**，吞吐量提升明显。

### 2.1 几个名词定义

- **目标模型（target model）**：那个大模型（Gemma4）。慢但准。
- **draft 模型（draft model）**：那个小模型（DSpark）。快但可能不准。
- **draft tokens**：draft 模型猜出来的候选 token。
- **接受（accept）**：大模型也认为这个 token 对，确认采用。
- **拒绝（reject）**：大模型不同意，丢弃这个 draft token。
- **接受率（acceptance rate）**：`接受数 / draft 总数`。这是投机解码的核心 KPI。

### 2.2 投机解码的标准流程

```
1. draft 模型猜 N 个 token:    [d0, d1, d2, d3]
2. 把 [id_last, d0, d1, d2, d3] 一次性喂给大模型
3. 大模型对每个位置算出"正确答案":
   id_last → t0 (大模型认为 d0 该是什么)
   d0      → t1
   d1      → t2
   d2      → t3
4. 比对：
   - 若 d0 == t0：接受 d0，看 d1 vs t1
   - 若 d1 == t1：接受 d1，看 d2 vs t2
   - 若 d2 != t2：拒绝 d2、d3，用 t2 作为下一个真值，从这里重新开始
5. 接受的 token 数 = 投机解码本步的产出 (1~N 个)
```

**最坏情况**：0 个接受（d0 就错了），相当于白跑一次 draft，损失一点时间。
**最好情况**：N 个全接受，1 次大模型 forward 出了 N+1 个 token。

---

## 第 3 章：DSpark 是什么？它和 EAGLE、DFlash 的关系

投机解码有几种实现，主要区别是"draft 模型怎么生成候选"：

| 方案 | draft 来源 | 特点 |
|---|---|---|
| **draft-simple** | 一个完整的小型 LLM（比如 Qwen2-0.5B） | 自己跑 forward，每步出 1 个 token |
| **EAGLE / EAGLE3** | 一个轻量"推测头"，吃目标模型中间层特征 | 每步出 1 个，递归猜 N 个 |
| **MTP** | 多 token 预测头，一次出 N 个 | 训练时按链式预测 |
| **DFlash** | 像 EAGLE3 但用 **block + mask** 一次性出 N 个 | 半自回归，block 内非 causal |
| **DSpark** | DFlash backbone + **Markov head**（在 logits 上加偏置） | 半自回归 + 离散马尔可夫修正 |

### 3.1 DFlash 的核心创新：block + mask

DFlash 不再像 EAGLE 那样"递归猜一个"，而是把整个 block 一次性丢进 draft：

```
draft batch:  [id_last, MASK, MASK, MASK, MASK, MASK, MASK]   (长度 = block_size, 默认 7)
```

- **id_last**：已确认的最后一个 token（"锚点"，anchor）。
- **MASK**：一个特殊 token（`mask_token_id`），表示"这个位置还没确定，待填"。
- draft 模型对整个 block 做**非 causal attention**（每个位置都能看到所有其他位置），一次性算出每个位置的 logits。

### 3.2 DSpark 比 DFlash 多了什么？

**Markov head**（马尔可夫头）。

DFlash 的 draft 是"独立预测每个位置"，假设各位置无关。但实际上，draft 内位置 `i` 的预测应该依赖位置 `i-1` 选了什么。DSpark 加了一个轻量修正：

```
logits_at_position_i += markov_w2 @ markov_w1[prev_token_at_position_i-1]
```

- `markov_w1`：shape `[R, n_vocab]`，把每个 token 映射到一个 R 维"上下文向量"（R = `markov_rank`，通常几十）。
- `markov_w2`：shape `[R, n_vocab]`，把"上下文向量"反映射回 vocab 维度。
- 所以 `markov_w2 @ markov_w1[prev]` 是一个 `[n_vocab]` 向量，按"上一个 token"给当前位置的 logits 加偏置。

这相当于一个**学习过的"二元语法模型"**（bigram），叠在 draft 的 neural logits 上。

### 3.3 "链式 argmax"是什么意思？

Markov head 的 prev_token 不能"采样"得到（那样要等采样器跑完才能算下一个），所以图里用的是 **argmax**：

```
prev_token_0 = anchor (id_last)
for i in 0..block_size-1:
    bias = markov_w2 @ markov_w1[prev_token]
    logits_i += bias
    prev_token = argmax(logits_i)     ← 这里是 greedy 的，不是采样
```

**这是个隐藏的不一致**：图内算 bias 时用的是 argmax 版本的前一个 token，但实际写出去的 draft token 是采样器采的。如果采样器开了 temperature，采到的 token 可能 ≠ argmax 选的 → bias 加在了"另一个 token 对应的行"上，对不上。

详见第 4 章 4.2。

---

## 第 4 章：开启 DSpark 时 Gemma 的完整推理流程

下面把每一步都拆开讲，**重点标出哪些环节影响接受率**。

### 4.1 process()：从目标模型抽特征，注入 draft 的 KV cache

**位置**：`common/speculative.cpp:1017-1123`

每次目标模型 forward 完之后，调用 `process()`：

```
A. 在目标模型里标记几个"extract 层"（target_layer_ids）
   llama_set_embeddings_layer_inp(ctx_tgt, layer_id, true)
   → 目标模型 forward 时，会把这些层的"输入嵌入"也吐出来

B. 把这些层的输出 interleave 拼起来：
   features[i, :] = [layer_A_input_embd, layer_B_input_embd, layer_C_input_embd]
   shape = [n_tokens, target_layer_ids_n * n_embd_tgt]

C. 把 features 喂给 DSpark 的 encoder：
   llama_encode(ctx_dft, features)
   → encoder 输出 n_embd_dec 维的"压缩特征"

D. 把 encoder 输出作为 embedding，喂给 DSpark 的 decoder（embd 模式）:
   llama_decode(ctx_dft, batch_inject)
   → decoder 把这些 embedding 的 K/V 写进自己的 KV cache
   → 现在 draft 的 KV cache 里有"目标模型看过的上下文"了
```

**为什么要这样**：draft 模型小，参数少，让它"看一眼"目标模型中间层的特征，可以补足它的理解能力。

**接受率风险点**（4.1）：

- **extract 层选错**：转换器写 `target_layer_ids = [i+1 for i in target_layer_ids]`（`gemma.py:810`），运行时也按 +1 解读。如果一边 +1 一边不 +1，抽错层。
- **抽取顺序错**：features 是 interleave 拼接，层顺序必须严格对应（`speculative.cpp:1070-1080`）。
- **多 ubatch prefill 漏抽**：长 prompt 切成多段 ubatch，每段都要跑一次 process。只跑最后一段 → 前 N-K 个 token 的 KV 是空的。
  - 现象：`LOG_WRN ... process() did not run on every prefill ubatch`（`speculative.cpp:1011`）

### 4.2 draft()：组装 mask block，跑 DSpark decoder

**位置**：`common/speculative.cpp:1247-1316`

```
A. 为每个 seq 组装 batch:
   [id_last, MASK, MASK, MASK, MASK, MASK, MASK]   (长度 = block_size, 默认 7)
   ↑ 这是 anchor-first 布局

B. llama_decode(ctx_dft, batch)
   → DSpark decoder 图（src/models/dspark.cpp:74-127）做这些事:
      1. 跑 transformer 主体，得到 base logits [n_vocab, n_tokens]
      2. 取出每个 block 的 anchor token (位置 0)
      3. 对 block 内每个位置 i:
         - 用 markov_w1[prev] 算上下文向量
         - 用 markov_w2 投回 vocab 维度
         - 加到 base logits[i] 上
         - argmax 得到下一个 prev
      4. 重排输出顺序，回填到 res->t_logits

C. 采样器在 biased logits 上采样:
   for i in 0..n_draft-1:
       common_sampler_sample(smpl, ctx_dft, beg + i, true)
       id = cur_p->data[0].id
       common_sampler_accept(smpl, id, true)
       result.push_back(id)
   → 得到 [d0, d1, d2, ...] 共 n_draft 个 draft token
```

**接受率风险点**（4.2）：

1. **anchor token 错位**：anchor 必须等于 `dp.id_last`（`speculative.cpp:1279`）。如果 id_last 不等于上一轮接受推进后的最后一个 token，整个 block 的 bias 全错。

2. **argmax 与采样器不一致**（**DSpark 最大的坑**）：
   - 图内 `prev = argmax(col)`（`dspark.cpp:116`）用的是 greedy
   - 实际 draft token 是采样器采的（`speculative.cpp:1305`）
   - 采样器开了 temperature/top_p → 采到的 ≠ argmax → bias 加错行
   - 训练时 DSpark 假设是 greedy，采样器越激进偏离越多
   - **排查**：临时把采样器设为 greedy（temperature=0），看接受率是否回升

3. **block 拼接后不是 block_size 整数倍**：
   - DSpark decoder 图里 `if (n_tok % bs != 0) return;`（`dspark.cpp:94`）
   - 多 seq 并发时，各 seq 的 n_draft 不同，拼接后可能不整除
   - 不整除时 **静默跳过 Markov bias**（base logits 直接出去），等于退化成 DFlash
   - **排查**：在 `speculative.cpp:1283` 加日志确认 `batch.n_tokens % block_size == 0`

### 4.3 目标模型校验：一次 batch forward

**位置**：`examples/speculative-simple/speculative-simple.cpp:216-234`

```
common_batch_clear(batch_tgt);
common_batch_add(batch_tgt, id_last, n_past++, {seq_id}, true);
for i in 0..draft.size()-1:
    common_batch_add(batch_tgt, draft[i], n_past + i, {seq_id}, true);

llama_decode(ctx_tgt, batch_tgt);   ← 大模型一次看 [id_last, d0, d1, ...]
llama_decode(ctx_dft, batch_tgt);   ← draft 模型也看一遍同样的 batch
```

注意：目标模型和 draft 模型都看了一遍**同一个 batch**。这是因为后续采样校验需要 draft 在这些位置的 logits。

**接受率风险点**（4.3）：
- **目标侧 causal_attn 必须是 true**，否则未来 token 被屏蔽，logits 与训练分布不符。
- **batch 大小**：如果 `n_ubatch < block_size + 1`，llama.cpp 内部会拆 batch，可能引发 KV cache 注入位置错乱。

### 4.4 接受判定：common_sampler_sample_and_accept_n

**位置**：`examples/speculative-simple/speculative-simple.cpp:249`

```cpp
auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);
```

这个函数做的事：

```
for i in 0..draft.size():
    t_i = sampler.sample(target_logits[i])
    if draft[i] == t_i:
        接受 draft[i]，继续看下一个
    else:
        不接受 draft[i] 及之后的所有
        把 t_i 作为新的"真值 token"，跳出循环
return 所有接受的 + 最后那个真值
```

返回的 `ids.size()` 至少是 1（哪怕全拒绝，也有 1 个真值 token）。

**接受率风险点**（4.4）：
- 这里的"接受"标准是**采样后 token 相等**，不是 logits 相等。所以即使 draft 和目标的 logits 排名一致，只要采样器在边界情况下抽到不同 token，也会判拒。

### 4.5 common_speculative_accept()：更新状态

**位置**：`common/speculative.cpp:2673-2700`

```cpp
void common_speculative_accept(common_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted) {
    impl->n_call_accept++;
    impl->n_acc_drafts++;
    for (i in 0..n_accepted-1) impl->n_acc_tokens_per_pos[i]++;
    impl->n_acc_tokens += n_accepted;
    impl->accept(seq_id, n_accepted, false);
    // 让其他并存的实现也收到通知（比如同时跑了 ngram-map）
    for (impl_other : other_impls) impl_other->accept(seq_id, n_accepted, true);
}
```

这是统计入口。`n_acc_tokens_per_pos[i]` 记录"第 i 个位置被接受了多少次"，用来分析"哪个位置最容易拒"。

**关键观察**：如果 `n_acc_tokens_per_pos[0]` 很低但 `[1]` `[2]` 高，说明 anchor 那一位置错（4.2 的 anchor 错位）。如果 `[0]` 高但越往后越低，说明 Markov 链式 argmax 偏离（4.2 的 argmax 不一致）。

### 4.6 接受率统计怎么算

**位置**：`common/speculative.cpp:144-158`

```cpp
size_t n_call_accept = 0;   // accept() 被调用的次数
size_t n_gen_drafts  = 0;   // 总共 draft 出了多少个 token
size_t n_acc_drafts  = 0;   // 多少次"有 >=1 个 token 被接受"
size_t n_acc_tokens  = 0;   // 总共接受了多少个 token
std::vector<size_t> n_acc_tokens_per_pos;  // 每个位置的接受次数
int64_t t_accept_us  = 0;   // 在 accept 上花的时间
```

**接受率公式**：

```
acceptance_rate = n_acc_tokens / n_gen_drafts
```

比如总共 draft 了 1000 个 token，接受了 600 个，接受率 = 60%。

**有效加速比**（speedup）则是另一个概念，不仅看接受率，还要看 draft 模型本身的耗时：

```
理想加速 = block_size 倍（全接受）
实际加速 ≈ 接受率 * block_size * (1 - draft 耗时占比)
```

---

## 第 5 章：Gemma 在 DSpark 路径里的家族特殊性

DSpark 不是 Gemma 独有（Qwen3 也有），但 Gemma4 有几个**家族特点**会影响接受率，必须特别处理。

### 5.1 转换器侧（`conversion/gemma.py:783`）

```python
class Gemma4DSparkModel(Gemma4Model):
    model_arch = gguf.MODEL_ARCH.DSPARK   # 和 Qwen3DSpark 共用同一个枚举

    def set_vocab(self):
        # 词表来自 --target-model-dir 指定的目标模型
        # 避免 draft 和目标 vocab 不一致
        Gemma4Model._set_vocab_gemma4(self, vocab_size)
        if mask_token_id:
            self.gguf_writer.add_mask_token_id(mask_token_id)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_target_layers([i+1 for i in target_layer_ids])
        self.gguf_writer.add_block_size(self.hparams.get("block_size", 7))
        self.gguf_writer.add_markov_rank(self.hparams.get("markov_rank", 0))

    def filter_tensors(cls, item):
        name = item[0]
        # 这些张量从 draft GGUF 里删除，运行时通过 ctx_other 共享目标模型的
        if name.endswith((
            "embed_tokens.weight",                # 用目标模型的 tok_embd
            "layer_scalar",                       # Gemma 专属，draft 不带
            "pre_feedforward_layernorm.weight",   # sandwich norm，draft 不带
            "post_feedforward_layernorm.weight",  # sandwich norm，draft 不带
        )):
            return None
        ...

    def modify_tensors(self, data_torch, name, bid):
        # Gemma 的 K/V 可以共享
        if self.hparams.get("attention_k_eq_v") and name.endswith(".self_attn.k_proj.weight"):
            yield from super().modify_tensors(data_torch, name, bid)
            yield from super().modify_tensors(data_torch,
                name.replace(".k_proj.weight", ".v_proj.weight"), bid)
```

**小白理解**：
- draft 不带 `embed_tokens` 和 `lm_head`（省空间，共享目标模型的）
- draft 不带 Gemma4 专属的夹层 norm 权重（DFlash decoder 里这些是 `TENSOR_NOT_REQUIRED`，见 `src/models/dflash.cpp:59-61`）
- draft 的 K=V 共享：v_proj 权重直接复制 k_proj 的（前提是配置允许）

### 5.2 推理图侧

- DSpark 的 encoder 和 decoder 都直接复用 DFlash 的图（`src/models/dspark.cpp:42-76`）
- DFlash decoder 在 `src/models/dflash.cpp:117-291`
- 因为那些 sandwich norm 是 NOT_REQUIRED，推理时走 else 分支跳过

```cpp
// src/models/dflash.cpp:264-269
if (layer.out_scale != nullptr) {
    inpL = ggml_mul(ctx0, cur, layer.out_scale);
} else {
    inpL = cur;   // ← Qwen 和 Gemma DSpark 都走这里
}
```

---

## 第 6 章：接受率不足的 8 个可疑环节（重点）

下面把所有可能让接受率掉的因素按可能性排序，**每一项都给出代码位置、机理、排查方法**。

### 6.1 【最高危】draft 和目标 vocab 不一致

**位置**：`common/speculative.cpp:78-122`

**机理**：
- 接受判定是"token id 相等"（`draft[i] == t_i`）
- 如果 draft 的词表和目标不一样，draft 输出的 token id 在目标里指代的是完全不同的字
- 那就**永远不会接受**

**症状**：
- 启动日志看到 `SPC_WRN: draft model vocab type must match target model`
- 或 `SPC_DBG: token N content differs - target 'X', draft 'Y'`
- 接受率恒为 0

**排查**：
- 检查 draft GGUF 是否用 `--target-model-dir` 正确生成（`conversion/gemma.py:793-800` 会借用目标模型的 tokenizer）
- 用 `gguf-py` 工具 dump 两个模型的 vocab，对比前 1000 个 token 的字符串

### 6.2 【最高危】目标模型 extract 层抽取错位

**位置**：`common/speculative.cpp:984-988, 1070-1080`；转换器 `gemma.py:810`

**机理**：
- 转换时 `target_layer_ids` 写入 GGUF 时已经 `+1`（`gemma.py:810`）
- 运行时 `llama_set_embeddings_layer_inp(ctx_tgt, target_layer_ids[k], true)` 打开对应层的开关
- 抽取的张量必须严格按层顺序 interleave（`speculative.cpp:1070-1080`）

**典型故障**：
| 故障 | 现象 |
|---|---|
| `target_layer_ids` 没按层号顺序 | draft 输出看似合理但接受率 0~5% |
| `+1` 约定不一致（一边 +1 一边 0-based） | draft token 与目标完全不相关 |
| 抽取层是 post-norm 输出而非 input | encoder 输入分布偏移，draft 永远发散 |
| 多 ubatch prefill 只抽了最后一段 | prefill 后前几个 token 接受率异常低 |

**排查**：
- 看 `LOG_INF: n_extract=%u, block_size=%d, mask_token_id=%d`（`speculative.cpp:964`）是否合理
- 在 `speculative.cpp:1076` 加临时日志，打印 layer index 与 token 对应关系
- dump GGUF 里 `dspark.target_layers` 整型数组，与目标模型 `config.json` 对比

### 6.3 【最高危】Markov argmax 与采样器口径不一致

**位置**：`src/models/dspark.cpp:106-118`（图内 argmax）；`common/speculative.cpp:1304-1314`（采样器）

**机理**：
- DSpark decoder 图内：`prev = ggml_argmax(col)`（`dspark.cpp:116`）用 greedy
- 实际写出去的 draft token：`common_sampler_sample(smpl, ctx_dft, beg + i, true)`（`speculative.cpp:1305`）采样
- 采样器开了 temperature → 采到的可能 ≠ argmax
- Markov bias 是按"前一个 token"索引的：如果采到的 ≠ argmax，下一个位置的 bias 加在了另一个 token 对应的行上
- 训练时 DSpark 假设是 greedy，采样器偏离越远接受率越低

**排查**：
- 临时把采样器设为 greedy（temperature=0, top_k=1），看接受率是否回升
- 若回升 → 采样器与训练分布不匹配，应改用更接近 greedy 的参数
- 若不回升 → 问题在别处

### 6.4 【高危】anchor token 错位

**位置**：`common/speculative.cpp:1279`

**机理**：
- 每个 block 第一个位置是 anchor，必须是 `dp.id_last`
- `dp.id_last` 应当等于上一轮接受推进后的最后一个 token
- 如果某次回滚时 `ckpt.pos_max` 不对，下一轮 anchor 就错位

**症状**：偶发性接受率塌陷，重启后好转

**排查**：
- 在 `speculative.cpp:1279` 加日志确认 `dp.id_last == 上次 accept 推进后的最后 token`
- 在 `dspark.cpp:107` 打出 `prev` 的前若干值

### 6.5 【高危】block 拼接后不是 block_size 整数倍

**位置**：`src/models/dspark.cpp:94-96`；`common/speculative.cpp:1278`

**机理**：
- DSpark decoder 图：`if (n_tok % bs != 0) return;`（`dspark.cpp:94`）
- 多 seq 并发时，各 seq 的 n_draft 不同，拼接后可能不整除
- 不整除时**静默跳过 Markov bias**，base logits 直接出去，退化成 DFlash
- 用户以为开了 DSpark，实际效果是 DFlash

**排查**：
- 单 seq 测试，让 `n_max == block_size`
- 多 seq 场景在 `speculative.cpp:1283` 加日志确认 `batch.n_tokens % block_size == 0`
- 看 `LOG_WRN: requested draft size %d exceeds the trained DFlash block size %d`（`speculative.cpp:968`）

### 6.6 【中危】Gemma4 特有的 norm 偏移、K=V 共享、iSWA

**位置**：`conversion/gemma.py:632-634`；`src/models/dflash.cpp:59-61, 264-269`

**机理**：
- **norm +1 偏移**：Gemma3/4 的 RMSNorm 是 `output * (1 + w)`，转换时对 `norm.weight` 做 `+1`（`gemma.py:125`）。若 DSpark 检查点残留了未偏移的 norm 权重，decoder 输出会偏 ~1 倍
- **K=V 共享**：`attention_k_eq_v` 为真时 v_proj 是 k_proj 的副本。若运行时 hparams 没传 `attention_k_eq_v` 但权重按共享存的，会出错
- **iSWA（interleaved sliding window attention）**：DFlash 把 SWA 层的 K/V 路由到滑动子缓存（`dflash.cpp:130-134, 168-175`）。若 `is_swa_impl` 数组长度或顺序与目标不一致，draft 看到的上下文错位

**排查**：
- 用 `llama-print-tensors` 或加载日志确认 `norm.weight` 均值 ≈ 1.x（已偏移）而非 ≈ 0.x（未偏移）
- 单独跑 draft model 非投机模式，看输出是否合理
- 比对 draft 与目标在同一 prompt 下的 logits 分布（KL 散度）

### 6.7 【中危】causal attention 没正确设置

**位置**：`common/speculative.cpp:990-991`

```cpp
llama_set_embeddings_nextn(ctx_dft, true, /*masked=*/true);
llama_set_causal_attn(ctx_dft, false);  // DFlash needs non-causal attention
```

**机理**：
- draft batch 是 `[anchor, MASK, MASK, ...]`，每个位置要看整个 block（包括"未来"的 mask），所以必须关 causal
- 目标侧 `ctx_tgt` 必须保持 causal_attn = true
- 如果某次重构不小心把这两行删了 → 接受率掉到 0，但不一定报错

**排查**：
- `llama_print_context` 输出里 `causal_attn = 0` 必须为 0 for ctx_dft
- `causal_attn = 1` 必须为 1 for ctx_tgt

### 6.8 【低危】checkpoint 部分接受回滚路径

**位置**：`examples/speculative-simple/speculative-simple.cpp:258-281`

**机理**：
- 部分接受时 restore checkpoint，draft KV cache 也回滚
- 若回滚时 `ckpt.pos_max` 不对，下一轮 draft 的 anchor 与目标 KV 位置错位 → 连锁错误

**症状**：偶发性接受率塌陷

**排查**：单跑非 checkpoint 模式（如果有），看是否消失

---

## 第 7 章：调试接受率的推荐顺序

```
1. 用 examples/speculative-simple 而不是 server
   → 把多 seq / 多 slot 因素排除

2. 采样器设为 greedy（temperature=0, top_k=1）
   → 看基线接受率

3. 打开 LOG_INF，关注:
   - draft GGUF 的 dspark.block_size / dspark.target_layers / dspark.markov_rank
   - ctx_dft 加载后的 hparams.n_dspark_block
   - 每次 draft() 的 batch.n_tokens 与 n_blocks
   - 每次 accept() 的 n_accepted 分布

4. 比对 draft logits 与 target logits 的 top-1 一致性（不开采样器，直接 argmax）:
   - 一致率高 → 问题在采样器（6.3）
   - 一致率低 → 问题在 encoder/decoder（6.2, 6.6, 6.7）

5. 单独跑 draft model 不开投机，生成一段文本:
   - 流畅 → draft 本身没问题，问题在 KV 注入或 extract 层
   - 不流畅 → draft 权重/norm 偏移/K=V 共享有问题
```

---

## 第 8 章：症状 → 原因 速查表

| 症状 | 可能原因（按可能性排序） |
|---|---|
| 接受率恒为 0 | vocab 不一致（6.1）；target_layer_ids 错（6.2）；causal_attn 没关（6.7） |
| 接受率 < 10%，draft 输出乱码 | norm 偏移没做（6.6）；extract 层错位（6.2） |
| 接受率 < 30%，draft 输出合理 | 采样器与 argmax 不一致（6.3）；anchor 错位（6.4） |
| 接受率时高时低 | 拼接不整除（6.5）；checkpoint 回滚错（6.8） |
| 多 seq 比单 seq 接受率低很多 | batch 拼接后 `n_tok % bs != 0`（6.5）；iSWA 路由错（6.6） |
| 长 prompt 后接受率掉 | process() 漏 ubatch（6.2）；KV cache 容量不足触发驱逐 |
| Greedy 时接受率高，采样时低 | 采样器与训练分布不匹配（6.3） |
| 接受率随生成长度逐渐下降 | KV cache 累计偏差，可能与 iSWA 滑窗配置有关（6.6） |

---

## 第 9 章：关键代码索引

| 功能 | 文件:行 |
|---|---|
| DSpark decoder 图（Markov bias） | `src/models/dspark.cpp:74-127` |
| DSpark encoder 图（= DFlash encoder） | `src/models/dspark.cpp:42-43` |
| DFlash decoder 图（KV 注入 + 噪声块） | `src/models/dflash.cpp:117-291` |
| DFlash encoder 图（特征融合） | `src/models/dflash.cpp:98-112` |
| DSpark draft batch 构造 | `common/speculative.cpp:1247-1316` |
| DSpark KV 注入流程 | `common/speculative.cpp:1017-1123` |
| 接受/校验主循环（示例） | `examples/speculative-simple/speculative-simple.cpp:180-290` |
| 接受统计字段 | `common/speculative.cpp:144-158` |
| Gemma4 DSpark 转换 | `conversion/gemma.py:783-835` |
| Qwen3 DSpark 转换（对比用） | `conversion/qwen.py:678-725` |
| vocab 一致性检查 | `common/speculative.cpp:78-122` |

---

## 第 10 章：一句话总结

**不开 DSpark**：大模型自己一步一步走，每步 1 个 token，慢但稳。

**开 DSpark**：让 DSpark 小模型一次猜 7 个 token，大模型一次校验 7 个，接受几个产出几个。接受率越高，加速越多。

**接受率上不去时**：按"vocab → extract 层 → argmax/采样器 → anchor → block 整除 → Gemma 专属 norm/KV/iSWA → causal_attn → checkpoint 回滚"的顺序排查，**90% 的接受率问题在前三项**。
