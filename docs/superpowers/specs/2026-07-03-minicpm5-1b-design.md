# MiniCPM5-1B 设计说明

日期：2026-07-03  
主题：为 `aha` 添加 MiniCPM5-1B 推理支持

## 1. 背景与目标

当前 `aha` 已支持多种模型家族，并通过统一的 `WhichModel`、`LoadSpec`、`GenerateModel`、CLI `run/serv`、以及 OpenAI 风格接口暴露推理能力。现需新增 **MiniCPM5-1B** 支持，已知本地可用权重为：

- Safetensors：`D:\model_download\MiniCPM5-1B`
- GGUF：`D:\model_download\MiniCPM5-1B-GGUF`

本次目标：

1. 在 `aha` 中新增 `minicpm5-1b` 模型标识。
2. 支持 **Safetensors** 与 **GGUF** 两种加载方式。
3. 支持 CLI `run` / `serv` / 根命令兼容模式的加载。
4. 支持 OpenAI 风格 chat completion 与 stream completion。
5. 尽量复用项目中已验证的通用组件，但保持 **MiniCPM5 独立 family**，不与 `qwen3_5` 绑定。

## 2. 已确认约束

### 用户确认

- 支持格式：**GGUF + Safetensors**
- 架构归属：**新建独立 family**

### 代码库约束

- 现有多格式模型通过 `LoadSpec` + `ArtifactKind` 接入。
- `run_target_model_with_spec` 负责优先走新式多格式路径。
- `registry::load_model_from_spec` 负责服务端/统一加载路径。
- `ChatTemplate` 已支持从 `metadata.enable_thinking` 注入 `enable_thinking` 变量。
- 现有 GGUF 基础设施已支持：
  - 从 GGUF metadata 构造 tokenizer
  - 从 GGUF metadata 读取 `chat_template`
  - 加载量化线性层、RMSNorm、MLP 等通用组件

## 3. 模型事实与架构判断

根据 `D:\model_download\MiniCPM5-1B\config.json` 与 README：

- `architectures = ["LlamaForCausalLM"]`
- `model_type = "llama"`
- `hidden_size = 1536`
- `num_hidden_layers = 24`
- `num_attention_heads = 16`
- `num_key_value_heads = 2`
- `head_dim = 128`
- `intermediate_size = 4608`
- `rms_norm_eps = 1e-6`
- `rope_theta = 5000000`
- `torch_dtype = "bfloat16"`
- `vocab_size = 130560`
- `eos_token_id = [1, 130073]`

因此 MiniCPM5-1B 不是 `Qwen3.5` 那种专有结构，而是 **标准 LlamaForCausalLM** 变体。设计上应：

- **不复用 `qwen3_5` 的专有模型实现**；
- **优先复用 `src/models/common` 中的 Llama 通用实现与注意力/MLP 基元**。

## 4. 方案选择

### 方案 A：直接复用 `qwen3_5` family

优点：
- 现成多格式框架完整。

缺点：
- 与 MiniCPM5 实际架构不符。
- 会把 Llama 类模型错误耦合到 Qwen3.5 专用实现。
- 后续维护成本高，语义上也不清晰。

结论：**不采用**。

### 方案 B：新建 `minicpm5` family，但底层复用 common Llama 组件（推荐）

优点：
- family 隔离清晰，符合用户要求。
- Safetensors 路径可直接建立在 `common::LlamaForCausalLM` 之上。
- GGUF 路径可复用 `common::gguf` 的量化组件，避免重复造轮子。
- 改动范围集中，便于测试。

缺点：
- 仍需补一套 `minicpm5::{config, model, generate}` 封装。
- GGUF 路径需要补一层本地的 Llama-quantized 装配逻辑。

结论：**采用该方案**。

### 方案 C：先只接入 Safetensors，再后补 GGUF

优点：
- 实现最短。

缺点：
- 与用户已确认范围不符。
- 无法直接使用当前已有的 GGUF 文件。

结论：**不采用**。

## 5. 目标设计

## 5.1 新增 family 与模块布局

新增目录：

- `src/models/minicpm5/mod.rs`
- `src/models/minicpm5/config.rs`
- `src/models/minicpm5/model.rs`
- `src/models/minicpm5/generate.rs`
- `src/exec/minicpm5.rs`

职责划分：

### `config.rs`
负责反序列化 safetensors 目录下的 `config.json`，至少包含：

- `bos_token_id`
- `eos_token_id`
- `pad_token_id`
- `hidden_act`
- `hidden_size`
- `intermediate_size`
- `max_position_embeddings`
- `num_attention_heads`
- `num_hidden_layers`
- `num_key_value_heads`
- `head_dim`
- `rms_norm_eps`
- `rope_theta`
- `torch_dtype`
- `tie_word_embeddings`
- `use_cache`
- `vocab_size`

### `model.rs`
提供 MiniCPM5 的模型装配逻辑：

- `MiniCPM5Model::new_from_vb(...)`：Safetensors 原生路径
- `MiniCPM5Model::new_from_gguf(...)`：GGUF 量化路径
- `forward(...)`：输出下一 token logits
- `clear_kv_cache()`：清理增量推理缓存

### `generate.rs`
提供统一推理入口：

- `MiniCPM5GenerateModel::init_from_spec(...)`
- `init(...)`（safetensors）
- `init_from_gguf(...)`
- `generate(...)`
- `generate_stream(...)`

### `exec/minicpm5.rs`
为 CLI `run` 提供直接推理封装，模式上与 `exec/qwen3_5.rs` 和 `exec/minicpm4.rs` 保持一致。

## 5.2 `WhichModel` 与 artifact 设计

新增模型枚举：

- `WhichModel::MiniCPM5_1B`

相关元信息：

- `openai_model_id()` -> `"minicpm5-1b"`
- `owner()` -> `"OpenBMB"`
- `model_id()` -> `"OpenBMB/MiniCPM5-1B"`
- `model_type()` -> `"llm"`

artifact 策略：

- `supported_artifacts(MiniCPM5_1B)` -> `[Safetensors, Gguf]`
- `default_artifact(MiniCPM5_1B)` -> `Safetensors`

不新增单独的 `MiniCPM5Gguf` 枚举值。原因：

- 当前 artifact 系统已支持“同一模型 + 多种载体格式”。
- 用一个 `WhichModel` + `--artifact-format gguf --gguf-path ...` 更符合现有新架构。
- 避免枚举膨胀和 CLI 认知负担。

## 5.3 Safetensors 路径设计

Safetensors 路径采用“独立 family + 通用 Llama 内核”的方式：

- 读取 `config.json`
- 用 `VarBuilder::from_mmaped_safetensors(...)` 加载权重
- 使用 `common::LlamaForCausalLM` 组装模型

对应映射：

- `vocab_size`
- `hidden_size`
- `num_hidden_layers`
- `num_attention_heads`
- `num_key_value_heads`
- `head_dim`
- `intermediate_size`
- `hidden_act`
- `rms_norm_eps`
- `rope_theta`
- `attn_bias = false`
- `mlp_bias = false`
- `attn_pp_name = "self_attn"`
- `mlp_pp_name = "mlp"`
- `input_norm_pp_name = "input_layernorm"`
- `post_norm_pp_name = "post_attention_layernorm"`

这样可以与当前项目已有的 Llama 风格 block 保持一致，而无需为 MiniCPM5 再复制一套 Transformer 结构。

## 5.4 GGUF 路径设计

GGUF 路径单独在 `minicpm5/model.rs` 中实现量化版装配，不扩散到 `common` 做无关泛化。

设计原则：

1. **family 独立**：GGUF 逻辑仍由 `minicpm5` 自己拥有。
2. **低层复用**：复用 `common::gguf::{Gguf, QuantizedLinear, ProjKind, GateUpDownMLPGguf}`。
3. **命名贴近 llama.cpp 常见导出格式**：
   - `token_embd.weight`
   - `blk.{i}.attn_q.weight`
   - `blk.{i}.attn_k.weight`
   - `blk.{i}.attn_v.weight`
   - `blk.{i}.attn_output.weight`
   - `blk.{i}.attn_norm.weight`
   - `blk.{i}.ffn_norm.weight`
   - `blk.{i}.ffn_gate.weight`
   - `blk.{i}.ffn_up.weight`
   - `blk.{i}.ffn_down.weight`
   - `output_norm.weight`
   - `output.weight`（若不存在则回退到 `token_embd.weight`）

运行时：

- 先读取 `general.architecture`，并将其作为架构前缀；对 MiniCPM5 预期值是 `llama`。
- metadata 读取顺序必须确定化，优先尝试：
  - `general.dtype`（如有）
  - `{arch}.block_count`
  - `{arch}.attention.head_count`
  - `{arch}.attention.head_count_kv`
  - `{arch}.attention.key_length`
  - `{arch}.embedding_length`
  - `{arch}.feed_forward_length`
  - `{arch}.rope.freq_base`
  - `{arch}.attention.layer_norm_rms_epsilon`
- 如果 `general.architecture` 缺失，直接报错，不做静默猜测。
- 如果某个核心 metadata 缺失，错误信息中必须包含缺失 key 名称。
- 构建：
  - 量化 q/k/v/o 投影
  - RMSNorm
  - 量化 gate/up/down MLP
  - RoPE
  - KV cache

该实现只服务 MiniCPM5，不要求把所有 llama GGUF 都抽象成全局通用类。

## 5.5 tokenizer 与 chat template 设计

### Safetensors

直接沿用目录内已有资源：

- `ChatTemplate::init(path)`
- `TokenizerModel::init(path)`

收益：

- 自动读取 `tokenizer_config.json` / `chat_template.jinja`
- 自动兼容 `enable_thinking` 模式
- 不需要为 MiniCPM5 特判模板渲染逻辑

### GGUF

使用 `load_text_bootstrap_from_gguf(...)`：

- 从 GGUF metadata 恢复 tokenizer
- 从 `tokenizer.chat_template` 读取模板
- 从 `tokenizer.ggml.eos_token_id` 读取默认 eos（如果需要）

如果 GGUF 中 chat template 缺失，则报出明确错误，而不是静默退化。因为当前用户提供的是单一 `.gguf` 文件，不应要求再额外传 tokenizer 目录。

## 5.6 推理行为设计

### 输入与输出

MiniCPM5 本次仅支持 **文本 chat completion**。

不纳入本次范围：

- 图像输入
- 视频输入
- 音频输入
- ONNX

### 终止条件

MiniCPM5 配置中的 `eos_token_id` 为数组，因此生成时需要在任意 stop token 命中时停止，至少包括：

- `1`
- `130073`

### 采样策略

保持与现有文本模型一致：

- 使用 `get_logit_processor(...)`
- 支持 `temperature`、`top_p`、`seed`
- 默认 `max_tokens`：可沿用 `512` 或 `1024` 的项目现有风格

### Stream 行为

`generate_stream(...)` 设计为：

- 与 `MiniCPM4` 相同，逐 token 生成
- 复用当前“乱码 token 缓冲”策略，避免 UTF-8 半截输出
- 输出 `build_completion_chunk_response(...)`

### Tool-calling

MiniCPM5 模板支持 XML 风格 tool calling。本次设计采用折中方案：

- **stream 路径**：支持识别 `<tool_call>` / `</tool_call>`，行为对齐 `qwen3_5` 的 streaming 实现
- **非 stream 路径**：先保持为普通文本 completion，不额外做结构化 tool-call 解析

这样可以优先满足当前项目已有的流式工具调用兼容路径，同时不扩大本次实现范围。

## 5.7 CLI 与服务接入设计

### `src/models/mod.rs`
新增：

- `pub mod minicpm5;`
- `WhichModel::MiniCPM5_1B`
- `LISTED_MODELS` 注册
- `ModelInstance::MiniCPM5(...)`
- `GenerateModel for ModelInstance` 的 `generate/generate_stream` 分发

### `src/models/core/registry.rs`
新增：

- `ModelLoaderFamily::MiniCPM5`
- `resolve_model_loader_family(WhichModel::MiniCPM5_1B) -> MiniCPM5`
- `load_model_from_spec` 中调用 `MiniCPM5GenerateModel::init_from_spec(...)`

### `src/models/core/artifact.rs`
新增：

- `supported_artifacts(WhichModel::MiniCPM5_1B)`
- `default_artifact(WhichModel::MiniCPM5_1B)`

### `src/main.rs`
新增：

- `run_target_model_with_spec` 中支持 MiniCPM5
- `run_run` 中支持 MiniCPM5 fallback 路径
- `run_list` 自动展示新增模型
- `download` 子命令自动支持 safetensors 下载管理

### `src/exec/mod.rs` 与 `src/exec/minicpm5.rs`
新增 MiniCPM5 的 CLI 直接运行实现，行为与现有文本 LLM 保持一致。

## 6. 错误处理设计

需要显式处理以下错误场景：

1. **artifact 不匹配**
   - `minicpm5-1b` 不支持 ONNX
2. **Safetensors 目录缺失关键文件**
   - `config.json`
   - `tokenizer.json`
   - `chat_template` 相关文件
   - `*.safetensors`
3. **GGUF 文件非法**
   - 路径不存在
   - 非 `.gguf` 文件
   - 缺少 tokenizer metadata
   - 缺少 chat template metadata
   - 缺少核心张量/metadata
4. **推理初始化不完整**
   - `qwen3_5` 现有实现中常见的“runtime not initialized”风格错误信息，MiniCPM5 应保持同级别清晰度
5. **GGUF 命名不兼容**
   - 若张量命名与预期不符，应在加载时报具体缺失张量名，便于定位转换问题

## 7. 测试设计

新增测试文件：

- `tests/test_minicpm5_multi_format.rs`

覆盖内容：

### Safetensors
- `init_from_spec` 可以正常初始化
- `generate` 至少返回 1 个 choice
- 如果本地目录不存在则打印 skip 信息并返回 `Ok(())`

### GGUF
- 能从 `.gguf` 文件初始化
- `generate` 至少返回 1 个 choice
- 如果找不到 `.gguf` 文件则 skip

### CLI 解析
在 `src/main.rs` 现有测试模块补充：

- `run --model minicpm5-1b --artifact-format gguf --gguf-path ...`
- `run --model minicpm5-1b --weight-path ...`
- `serv --model minicpm5-1b --artifact-format gguf --gguf-path ...`
- 根命令兼容模式 `--model minicpm5-1b ...`

### Registry / artifact
- `registry` 路由到 `MiniCPM5` family
- `supported_artifacts` 与 `default_artifact` 返回值正确

建议测试环境变量：

- `AHA_MINICPM5_SAFETENSORS_DIR`
- `AHA_MINICPM5_GGUF_PATH`

默认值可直接指向当前用户本地目录：

- `D:\model_download\MiniCPM5-1B`
- `D:\model_download\MiniCPM5-1B-GGUF\MiniCPM5-1B-Q4_K_M.gguf`

## 8. 非目标与范围边界

本次不做：

1. MiniCPM5 多模态输入支持
2. ONNX Runtime 支持
3. 通用 llama GGUF 抽象重构到 `common`
4. 对现有 `qwen3_5` / `minicpm4` family 的架构调整
5. 模型下载器对 GGUF 目录的自动管理

这些都可以作为后续独立迭代。

## 9. 实施顺序建议

1. 新增 `minicpm5/config.rs`
2. 新增 `minicpm5/model.rs`，先完成 safetensors 路径
3. 新增 `minicpm5/generate.rs`，先打通 chat completion
4. 接入 `models/mod.rs`、`artifact.rs`、`registry.rs`
5. 接入 `main.rs` 与 `exec/minicpm5.rs`
6. 再补 GGUF 路径
7. 最后补 multi-format tests 与 CLI parse tests

## 10. 最终设计结论

本次采用：

- **独立 MiniCPM5 family**
- **单模型枚举 + 多 artifact 格式**
- **Safetensors 默认、GGUF 可选**
- **Safetensors 复用 common Llama 内核**
- **GGUF 在 family 内局部实现量化 Llama 装配**
- **仅支持文本推理，stream 支持基础 tool-call 标签处理**

该设计与当前代码库的模块边界一致，能以较小改动完成 MiniCPM5-1B 接入，并为后续 ONNX / 多模态扩展保留清晰的演进空间。
