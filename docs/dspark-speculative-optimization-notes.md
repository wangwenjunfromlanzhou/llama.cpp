# DSpark 修改说明（小白版）

这份文档用尽量基础的方式解释本仓库里 DSpark 相关修改。

你可以先把 DSpark 理解成：

> 一个“小模型 / 草稿模型”先猜后面几个 token，大模型再检查这些 token 对不对。如果猜对，就能省时间；如果猜错，就回退到大模型正常生成。

这里的修改分两部分：

1. 已提交的 `3544db3`：让 llama.cpp 支持 Qwen3 DSpark。
2. 当前未提交修改：在前面基础上继续支持 Gemma4 DSpark。

---

## 1. 先理解几个基础概念

### 1.1 token 是什么

大模型不是直接一个字一个字生成文本，而是生成 token。

例如一句话：

```text
你好，世界
```

可能会被 tokenizer 切成几个 token。模型每次预测下一个 token。

### 1.2 target model 是什么

`target model` 就是最终负责输出质量的大模型。

它比较大、比较准，但每生成一个 token 成本高。

### 1.3 draft model 是什么

`draft model` 是小一点或特殊结构的草稿模型。

它先快速猜几个 token：

```text
draft: A B C D
```

然后 target model 检查：

```text
A 对，B 对，C 错
```

那就接受 `A B`，从 `C` 开始重新生成。

### 1.4 speculative decoding 是什么

speculative decoding 就是“草稿模型先猜，大模型验收”。

如果 draft 猜得准，就能一次接受多个 token，从而加速。

如果 draft 猜得不准，反而会变慢，因为多做了 draft 工作。

### 1.5 GGUF 是什么

GGUF 是 llama.cpp 使用的模型文件格式。

转换脚本会把 HuggingFace / ModelScope 上的模型权重转换成 GGUF，llama.cpp runtime 再加载 GGUF。

---

## 2. DSpark 是什么

DSpark 可以理解成一种特殊的 draft model。

它不是完全独立从零生成，而是依赖 target model 的中间层信息。

整体流程大概是：

```text
target model 运行一部分
        ↓
抽取若干层 hidden state
        ↓
送给 DSpark draft model
        ↓
DSpark 猜后面几个 token
        ↓
target model 检查这些 token
```

DSpark 的实现复用了 DFlash。

简单说：

- DFlash 负责从 target 抽特征、送进 draft model；
- DSpark 在 DFlash 基础上加了一个 Markov head；
- Markov head 用来让一个 block 内的多个 draft token 之间产生依赖。

---

## 3. Commit `3544db3` 做了什么

提交 `3544db3` 的标题是：

```text
Add DSpark speculative decoding support
```

它完成了 Qwen3 DSpark 的基础支持。

### 3.1 新增 `dspark` 模型架构

llama.cpp 需要知道“这是什么模型”。

所以这次新增了一个架构名：

```text
dspark
```

涉及的主要文件：

```text
gguf-py/gguf/constants.py
src/llama-arch.h
src/llama-arch.cpp
src/llama-model.cpp
src/models/models.h
```

这些修改的作用是：

1. 转换 GGUF 时知道有一种模型叫 DSpark。
2. 加载 GGUF 时能识别 `general.architecture = dspark`。
3. runtime 能创建对应的 `llama_model_dspark` 对象。

### 3.2 DSpark 继承 DFlash

代码里 DSpark 是这样定义的：

```cpp
struct llama_model_dspark : public llama_model_dflash
```

小白理解：

> DSpark 不是从零写一个全新模型，而是在 DFlash 的基础上加功能。

它复用了 DFlash 的：

- target hidden state 抽取；
- draft encoder；
- draft KV cache 注入；
- block 形式的 draft 推理。

DSpark 自己新增的是 Markov head。

### 3.3 新增 DSpark 的 GGUF 信息

DSpark GGUF 里需要保存一些额外信息。

主要有：

| 信息 | 小白解释 |
|---|---|
| `dspark.block_size` | 一次让 draft model 猜多少个位置 |
| `dspark.markov_rank` | Markov head 内部使用的低秩维度 |
| `dspark.target_layers` | 从 target model 哪些层抽 hidden state |

这些信息写进 GGUF 后，runtime 才知道应该怎么运行 DSpark。

### 3.4 新增 Markov head 权重

DSpark 新增了几个 tensor：

| tensor | 作用 |
|---|---|
| `markov_w1` | 根据上一个 token 找到 Markov 特征 |
| `markov_w2` | 把 Markov 特征变成 logits bias |
| `conf_proj` | 可选 confidence head |

你可以把 Markov head 理解成：

> 它根据“上一个 token 是什么”，调整“下一个 token 更可能是什么”。

### 3.5 Qwen3DSpark 转换支持

`conversion/qwen.py` 新增了：

```python
@ModelBase.register("Qwen3DSparkModel")
class DSparkModel(Qwen3Model):
    model_arch = gguf.MODEL_ARCH.DSPARK
```

这表示：

> 如果模型配置里写着 `Qwen3DSparkModel`，转换脚本就用这个类来转 GGUF。

这个转换类主要做几件事：

1. 要求用户传 `--target-model-dir`。
2. 从 target model 读取 tokenizer。
3. 写入 `mask_token_id`。
4. 写入 `block_size`。
5. 写入 `target_layer_ids`。
6. 写入 `markov_rank`。
7. 丢掉 `embed_tokens.weight` 和 `lm_head.weight`。

为什么丢掉 embedding 和 lm_head？

因为 Qwen3 DSpark 的这些权重和 target model 是一样的，没必要在 draft GGUF 里重复存一份。

### 3.6 runtime 怎么跑 DSpark

核心文件：

```text
src/models/dspark.cpp
```

它做了两件重要的事。

第一，加载 DSpark 参数和权重：

```cpp
block_size
markov_rank
markov_w1
markov_w2
conf_proj
```

第二，在 decoder graph 里给 logits 加 Markov bias。

简化公式：

```text
新的 logits = 原始 logits + Markov bias
```

Markov bias 来自上一个 token。

### 3.7 block 和 anchor 是什么

DSpark 一次不是只看一个 token，而是看一个 block。

比如 `block_size = 7`，输入可能是：

```text
[id_last, mask, mask, mask, mask, mask, mask]
```

其中：

- `id_last` 是已经确认的最后一个 token；
- 它叫 anchor；
- 后面的 `mask` 是让 DSpark 去猜的位置。

DSpark 会输出多个 draft token。

注意：position 0 输出的是“第一个 draft token 的分布”，不是 anchor 自己。

### 3.8 新增 `draft-dspark`

`common/speculative.cpp` 新增了一个 speculative 类型：

```text
draft-dspark
```

使用时可以选择 DSpark 草稿模式。

它的流程是：

```text
1. 准备 anchor-first block
2. 调用 draft model decode
3. 从 logits 中拿到 draft token
4. 交给 target model 验证
```

DSpark 的 speculative 实现继承 DFlash，所以它复用了 DFlash 的大部分流程，只重写了 draft token 的生成方式。

---

## 4. 当前未提交修改做了什么

当前未提交修改主要是：

> 在已有 Qwen3 DSpark 支持上，新增 Gemma4 DSpark 支持。

### 4.1 注册 `Gemma4DSparkModel`

`conversion/__init__.py` 增加了：

```python
"Gemma4DSparkModel": "gemma"
```

意思是：

> 如果模型配置里写着 `Gemma4DSparkModel`，转换脚本去 `conversion/gemma.py` 找对应转换逻辑。

### 4.2 新增 Gemma4DSparkModel 转换类

`conversion/gemma.py` 新增：

```python
@ModelBase.register("Gemma4DSparkModel")
class Gemma4DSparkModel(Gemma4Model):
    model_arch = gguf.MODEL_ARCH.DSPARK
```

小白理解：

> Gemma4 DSpark 仍然输出 `dspark` GGUF，只是转换时复用 Gemma4 的转换逻辑。

它继承的是 `Gemma4Model`，这样可以少写很多重复代码，比如：

- Gemma4 tokenizer 处理；
- Gemma4 RoPE 参数；
- Gemma4 attention 参数；
- Gemma4 其他 metadata。

### 4.3 Gemma4 DSpark 也要求 target tokenizer

Gemma4DSparkModel 里要求：

```python
--target-model-dir
```

如果没传，会报错。

原因很简单：

> draft model 和 target model 必须使用同一套 tokenizer，否则 token id 对不上，验证就不可靠。

转换时会临时切到 target model 目录，读取 target tokenizer，再切回 draft model 目录。

### 4.4 写入 DSpark 必要参数

Gemma4DSparkModel 会先调用 Gemma4 原本的参数写入逻辑：

```python
super().set_gguf_parameters()
```

然后额外写入 DSpark 需要的参数：

```python
target_layer_ids
block_size
markov_rank
```

其中 `target_layer_ids` 会做 `+1`。

小白版解释：

> config 里写的是模型层编号，但 runtime 需要的是“抽取 layer input 的编号”，所以转换时统一加 1。

### 4.5 过滤一些不需要的 tensor

Gemma4DSparkModel 会过滤这些 tensor：

```text
embed_tokens.weight
layer_scalar
layer_scalar.weight
pre_feedforward_layernorm.weight
post_feedforward_layernorm.weight
```

最重要的是：

```text
embed_tokens.weight
```

它和 target model 共享，不需要重复放在 draft 模型里。

### 4.6 处理 `attention_k_eq_v`

Gemma4 DSpark config 里可能有：

```json
"attention_k_eq_v": true
```

意思是：

> K projection 和 V projection 使用同一份权重。

但是 llama.cpp 加载模型时通常希望 K 和 V 都有。

所以转换时如果看到：

```text
k_proj.weight
```

会额外复制出：

```text
v_proj.weight
```

这样 runtime 加载时就不会缺 V 权重。

### 4.7 DSpark 支持 optional output

当前修改还让 DSpark 支持可选的 output 权重。

`gguf-py/gguf/constants.py` 里给 DSpark 增加：

```python
MODEL_TENSOR.OUTPUT
```

`src/models/dspark.cpp` 里增加：

```cpp
output = create_tensor(..., TENSOR_NOT_REQUIRED);
```

意思是：

> 如果 DSpark GGUF 里有 `output.weight`，就加载；如果没有，也不报错。

这样可以兼容两种情况：

| 模型 | output 权重 |
|---|---|
| Qwen3 DSpark | 不带，复用 target |
| Gemma4 DSpark | 可以带，runtime optional 加载 |

---

## 5. 本地 Gemma4 DSpark 模型目录

当前工作区里有一个未跟踪目录：

```text
dspark_gemma4_12b_block7/
```

它是一个 Gemma4 DSpark draft 模型目录。

关键配置大概是：

```json
{
  "architectures": ["Gemma4DSparkModel"],
  "attention_k_eq_v": true,
  "block_size": 7,
  "markov_rank": 256,
  "mask_token_id": 4,
  "hidden_size": 3840,
  "num_hidden_layers": 5,
  "target_layer_ids": [5, 17, 29, 41, 46],
  "vocab_size": 262144
}
```

这些配置和前面的代码改动是对应的：

- `Gemma4DSparkModel` 对应新增的转换类；
- `block_size=7` 对应 DSpark block 长度；
- `markov_rank=256` 对应 Markov head 维度；
- `target_layer_ids` 对应从 target 抽哪些层；
- `attention_k_eq_v=true` 对应复制 K 到 V 的逻辑。

如果目录里有：

```text
model.safetensors.incomplete
```

说明权重还没下载完整，暂时不能直接转换完整模型。

---

## 6. 用一张流程图串起来

### 6.1 Qwen3 DSpark

```text
Qwen3DSparkModel
        ↓
conversion/qwen.py
        ↓
生成 dspark GGUF
        ↓
src/models/dspark.cpp 加载
        ↓
common/speculative.cpp 使用 draft-dspark
        ↓
draft 猜 token，target 验证
```

### 6.2 Gemma4 DSpark

```text
Gemma4DSparkModel
        ↓
conversion/gemma.py
        ↓
生成 dspark GGUF
        ↓
可选加载 output.weight
        ↓
src/models/dspark.cpp 加载
        ↓
common/speculative.cpp 使用 draft-dspark
        ↓
draft 猜 token，target 验证
```

---

## 7. 最后总结

### `3544db3` 解决的问题

它让 llama.cpp 认识并运行 DSpark：

- 新增 `dspark` 架构；
- 支持 Qwen3DSparkModel 转换；
- 加载 DSpark Markov head；
- 新增 `draft-dspark` speculative decoding。

### 当前未提交修改解决的问题

它让 Gemma4 DSpark 也能走同一套 DSpark 路径：

- 注册 `Gemma4DSparkModel`；
- 复用 Gemma4 转换逻辑；
- 从 target model 读取 tokenizer；
- 写入 DSpark 参数；
- 处理 `attention_k_eq_v`；
- 支持 optional `output.weight`。

一句话总结：

> 之前的提交让 Qwen3 DSpark 能用；当前修改让 Gemma4 DSpark 也能用，而且两者共用同一套 `dspark` runtime 和 `draft-dspark` 推测解码路径。
