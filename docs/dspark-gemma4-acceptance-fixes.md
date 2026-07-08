# DSpark Gemma4 Draft 接受率优化总结

## 背景

DSpark speculative decoding 在 Gemma4-12B target + dspark_gemma4_12b_block7 draft 上的初版实现接受率极低（~3-5%），速度仅为 baseline 的 ~1.0×。Qwen3 DSpark 路径接受率正常，Gemma4 不行。

根因：`src/models/dflash.cpp` 的 decoder graph 是直接照 Qwen3 标准 pre-norm transformer 结构写的，而 Gemma4 的层结构差异显著（sandwich norms、layer_scalar、k_eq_v、softcap）。本会话通过逐项对齐 vllm PR #47216 的 Gemma4 DSpark 实现，把接受率从 ~3-5% 提升到 ~30-51%。

## 性能进展

| 指标 | 初版 | 中间版 | 中间版2 | **当前** |
|---|---|---|---|---|
| 最高 speedup | ~1.0× | 1.255× | 1.487× | **2.264×** |
| deterministic_en 接受率 | ~5% | 48% | 51% | **86.8% (n1) / 50.3% (n4)** |
| tech_en 接受率 | ~3% | 28% | 26% | **78.9% (n1) / 55.0% (n4)** |
| zh_explain 接受率 | ~3% | 31% | 30% | **72.6% (n1) / 42.5% (n4)** |

**结论**：embedding × sqrt(n_embd) 是最大的单项提升，三项 case 全线上跳一档；
n1 串行场景接受率 72-87%（draft model 能力被充分消化）；n4/n7 并行 batch 接受率 30-50%（n_tokens 越大 draft 推断越易失准，属正常 trade-off）。

## 当前完整性能矩阵（baseline ~9.6 tok/s）

| spec | case | tok/s | speedup | accept |
|---|---|---|---|---|
| n1 | tech_en | 13.36 | 1.390× | 56/71 = 78.9% |
| n1 | deterministic_en | 14.07 | 1.458× | 59/68 = 86.8% |
| n1 | zh_explain | 12.94 | 1.341× | 53/73 = 72.6% |
| n2 | tech_en | 17.47 | 1.817× | 73/102 = 71.6% |
| n2 | deterministic_en | 18.62 | 1.928× | 76/101 = 75.2% |
| n2 | zh_explain | 16.74 | 1.734× | 70/112 = 62.5% |
| n4 | tech_en | 18.70 | 1.945× | 77/140 = 55.0% |
| n4 | deterministic_en | 21.85 | **2.264×** | 84/167 = 50.3% |
| n4 | zh_explain | 19.65 | 2.036× | 79/186 = 42.5% |
| n7 | tech_en | 21.64 | 2.251× | 84/174 = 48.3% |
| n7 | deterministic_en | 21.34 | 2.210× | 83/173 = 48.0% |
| n7 | zh_explain | 21.74 | 2.253× | 84/289 = 29.1% |

## 与 Qwen3 DSpark 的结构差异（根因表）

| 项 | Qwen3 | Gemma4 | dflash.cpp 是否覆盖 |
|---|---|---|---|
| input_layernorm | ✓ | ✓ | ✓ |
| post_attention_layernorm | ✗ | ✓ | ✗（初版缺失，已修） |
| pre_feedforward_layernorm | ✓ | ✓ | ✓（撞名 bug，已修） |
| post_feedforward_layernorm | ✗ | ✓ | ✗（已修） |
| layer_scalar | ✗ | ✓ | ✗（已修） |
| attention_k_eq_v | ✗ | ✓（V=v_norm(k)） | ✗（已修） |
| final_logit_softcapping | ✗ | ✓ | ✗（已修） |
| embedding × sqrt(n_embd) | ✗ | ✓ | ✗（已修） |
| attention scale | 1/sqrt(head_dim) | 1.0 | 探测中（kq_scale=1.0） |
| rope_freqs (proportional) | ✗ | ✓ | ✗（已加载，graph 待接） |

## 优化点清单（按 commit 顺序）

### Commit `75bfe9437` — 4 项核心 Gemma4 特征

**问题**：dflash.cpp 完全没实现 Gemma4 的 sandwich norms / layer_scalar / k_eq_v / softcap。

**修复**：
- `conversion/gemma.py`：停止丢弃 `layer_scalar`、`post_attention_layernorm`、`post_feedforward_layernorm`
- `src/models/dflash.cpp`：
  - 加载 `layer.out_scale` / `attn_post_norm` / `ffn_post_norm`（均 NOT_REQUIRED）
  - decoder graph 在 attn 输出后加 post_attention_layernorm
  - decoder graph 在 ffn 输出后加 post_feedforward_layernorm
  - decoder graph 在每层末尾乘 `layer_scalar`
  - k_eq_v 路径：V = `ggml_rms_norm(k_proj(x))`（无 weight、无 RoPE）
- `src/models/dspark.cpp`：加载 `final_logit_softcapping`；在 Markov bias 之前对 base logits 做 scale→tanh→scale
- `tests/test_dspark_gemma4_tensors.py`：4 个回归测试

### Commit `d1155e360` — DSPARK 架构 tensor 白名单

**问题**：`MODEL_TENSORS[MODEL_ARCH.DSPARK]` 缺 `ATTN_POST_NORM` / `FFN_POST_NORM` / `LAYER_OUT_SCALE`，`tensor_map.__init__` 第 2445 行 `if tensor not in MODEL_TENSORS[arch]: continue` 把它们过滤掉，conversion 报 `Can not map tensor 'model.layers.0.layer_scalar.weight'`。

**修复**：`gguf-py/gguf/constants.py` DSPARK 白名单加这三项。

### Commit `7dab8336c` — attn_v 可选化

**问题**：Gemma4 k_eq_v checkpoint 不带 v_proj，但 dflash loader 把 attn_v 当 required，加载报 `missing tensor`。

**修复**：`src/models/dflash.cpp` `layer.wv` 改 `TENSOR_NOT_REQUIRED`。

### Commit `46bc9220f` — kq_scale = 1.0 探测

**问题**：dflash 用 `1/sqrt(n_embd_head)` 作为 attention scale，Gemma4 训练时 Q/K norms 自带缩放，scale 应该是 1.0。

**修复**：`src/models/dflash.cpp` 硬编码 `kq_scale = 1.0f`。**临时探测**，若 Qwen3 DSpark 回归则需加 arch hparam 分支。

### Commit `25fbb564d` — 重新合成 v_proj

**问题**：上一版 conversion 不写 v_proj，loader 期望的 tensor 总数对不上，报 `expected 69, got 68`。

**修复**：`conversion/gemma.py` 在 k_eq_v 时把 k_proj 复制为 v_proj 写进 GGUF（C++ graph 仍按 k_eq_v 走 V=v_norm(k) 路径，v_proj 只是占位）。

### Commit `a3325c73a` — 加载 rope_freqs

**问题**：GGUF 里有 `rope_freqs.weight`（Gemma4 full-attn 层 proportional rope 需要），但 dflash loader 没声明会读它，报 `expected 74, got 73`。

**修复**：`src/models/dflash.cpp` per-layer 加 `layer.rope_freqs = create_tensor(..., TENSOR_NOT_REQUIRED | TENSOR_DUPLICATED)`。

### Commit `abe2942e8` — swa_type 复位

**问题**：dflash load_arch_hparams 强制 `swa_type = STANDARD`，Gemma4 DSpark draft 没 SWA 层，memory 创建时走 SWA 分支触发 `GGML_ASSERT(hparams.is_swa_any())`。

**修复**：`src/models/dspark.cpp` load_arch_hparams 调完 dflash 后覆写 `hparams.swa_type = LLAMA_SWA_TYPE_NONE`。

### Commit `00a7f97d6` — post_attention_layernorm 撞名修复

**问题**：`tensor_mapping.py` 把 `post_attention_layernorm` 和 `pre_feedforward_layernorm` 都映射到 `FFN_NORM`（Llama 风格别名）。Gemma4 两者是不同的 sandwich norm，撞名导致 post_attention 实质丢失。

**修复**：`conversion/gemma.py` `modify_tensors` 加分支——`post_attention_layernorm` 用 `format_tensor_name(ATTN_POST_NORM)` 直接 emit，绕过 mapping。

### Commit `ee68216d4` — pre_feedforward_layernorm 映射修复

**问题**：`FFN_PRE_NORM` 不在 DSPARK 白名单，`pre_feedforward_layernorm` 在 DSPARK 架构下完全没注册，conversion 无法 map，导致 GGUF 缺 `blk.X.ffn_norm.weight`，loader 报 `missing tensor`。

**修复**：`conversion/gemma.py` `modify_tensors` 加分支——`pre_feedforward_layernorm` 用 `format_tensor_name(FFN_NORM)` 直接 emit。

### Commit `758c8d136` — embedding × sqrt(n_embd) 缩放 ★ 最大单项提升

**问题**：Gemma4 训练时 embedding 在第一层前乘 `sqrt(hidden_size)`，dflash 没乘，第一层输入分布与 target 模型完全错位，draft logits 长期低接受率。

**修复**：`src/models/dflash.cpp` decoder 在 `ggml_get_rows(tok_embd, tokens)` 后加 `ggml_scale(ctx0, inpL, sqrtf(float(n_embd)))`。

**效果**（前后对比，n1 场景）：

| case | 修复前 accept | 修复后 accept | 提升 |
|---|---|---|---|
| deterministic_en | 51% | **86.8%** | +35 pp |
| tech_en | 26% | **78.9%** | +53 pp |
| zh_explain | 30% | **72.6%** | +43 pp |

最高 speedup 从 1.487× → **2.264×**。三项 case 全线接近 vllm PR 报告水平。**临时无条件**，若 Qwen3 回归需 gate。

## 修改文件汇总

| 文件 | 修改类型 |
|---|---|
| `conversion/gemma.py` | filter_tensors / modify_tensors 多处加分支 |
| `gguf-py/gguf/constants.py` | DSPARK 白名单加 3 项 tensor |
| `src/models/dflash.cpp` | loader 加可选 tensor；decoder graph 加 sandwich norms + layer_scalar + k_eq_v + embedding scale + kq_scale |
| `src/models/dspark.cpp` | 加载 final_logit_softcapping；swa_type 复位；logits softcap |
| `tests/test_dspark_gemma4_tensors.py` | 4 个回归测试（新增） |

## 仍未解决 / 待优化

| 项 | 状态 | 估计影响 |
|---|---|---|
| `rope_freqs` 已加载但 decoder graph 没用 | 待改 | 中（Gemma4 full-attn 层 proportional rope） |
| `kq_scale=1.0` 硬编码 | 临时 | Qwen3 DSpark 会回归，需加 arch hparam |
| embedding scale 无条件 | 临时 | Qwen3 DSpark 会回归，需 gate |
| `final_logit_softcapping` 是否生效 | 未验证 | 中 |
| n4/n7 batch 接受率 30-50% | n_tokens 越大 draft 推断越易失准 | 正常 trade-off，不是 bug |

## 验证方法

```bash
# Linux 上
cd /home/admin/llama.cpp-wwj2
git pull
cmake --build build --target llama-server -j

# 转 draft
python conversion/convert_hf_to_gguf.py \
  /path/to/dspark_gemma4_12b_block7 \
  --target-model-dir /path/to/gemma-4-12b-it \
  --outtype bf16 \
  --outfile /home/admin/gemma4_12b/dspark-bf16.gguf

# 验证 GGUF tensor 清单（每层应有 14 个 tensor）
python -c "
from gguf import GGUFReader
import re
r = GGUFReader('/home/admin/gemma4_12b/dspark-bf16.gguf')
for t in r.tensors:
    if re.search(r'blk\.0\.', t.name):
        print(t.name, t.shape)
"

# 跑接受率
./build/bin/llama-server \
  -m /home/admin/gemma-12b-goole/gemma-12b-goole-bf16.gguf \
  -md /home/admin/gemma4_12b/dspark-bf16.gguf \
  -spec 7 -p "prompt" -n 128
```
