# LFM2.5-Embedding-350M 设计说明

日期：2026-07-05  
主题：为 `aha` 接入 `lfm2.5-embedding-350m` 本地 embedding 模型

## 1. 背景与目标

当前 `aha` 已具备统一的本地模型接入链路：

- `WhichModel` 负责模型枚举与元数据
- `LoadSpec` / `ArtifactKind` 负责多格式加载
- `ModelInstance` 负责统一能力分派
- CLI `run` / `serv` 与服务端 `/embeddings` 对外暴露能力

用户要求将本地模型目录：

- `D:\model_download\LFM2.5-Embedding-350M`

接入到当前项目，并明确选择：

1. 新增独立模型 ID，而不是复用现有 `lfm2.5-350m`
2. 做完整接入，而不是只做底层可加载
3. 内建 `query` / `document` prompt 语义，并允许调用方选择

本次目标：

1. 新增模型标识 `lfm2.5-embedding-350m`
2. 支持本地 `safetensors` 目录加载
3. 支持 CLI `run` 调用该模型输出 embedding
4. 支持服务端 `/embeddings` 调用该模型
5. 为 CLI 与服务端增加 `prompt_name=query|document` 可选参数
6. 补测试与文档，且不影响现有 `lfm2.5-350m` 文本生成路径

## 2. 已确认约束

### 用户确认

- 模型标识：新增独立模型 ID，不复用现有 `lfm2.5-350m`
- 接入范围：完整接入 CLI、`LoadSpec`、服务端、测试、文档
- prompt 策略：默认按 `document` 编码，同时允许显式指定 `query`

### 本地模型事实

从 `D:\model_download\LFM2.5-Embedding-350M` 目录可确认：

- 存在 `model.safetensors`
- 存在 `config.json`
- 存在 `tokenizer.json` 与 tokenizer 配置
- 存在 `config_sentence_transformers.json`
- 存在 `modules.json`
- 存在 `1_Pooling/config.json`

该目录不是当前仓库中现有 `lfm2.5-350m` 生成模型的普通目录结构，而是一个 `sentence-transformers` 风格的双向 embedding 模型目录。

### 代码库约束

- 现有 embedding 模型接入风格为：
  - `src/models/<family>/model.rs`
  - `src/models/<family>/generate.rs`
  - `src/exec/<family>.rs`
- `ModelInstance::embedding(...)` 目前只分派给：
  - `all-minilm-l6-v2`
  - `qwen3-embedding-*`
- 当前 `/embeddings` 只有 `input`，没有 query/document 语义参数
- 当前 `src/models/lfm2_5/model.rs` 是因果生成实现，使用 causal mask 与 cache

## 3. 模型事实与架构判断

根据本地 `config.json`、`config_sentence_transformers.json`、`modules.json` 与 `1_Pooling/config.json`：

- `architectures = ["Lfm2BidirectionalModel"]`
- `model_type = "lfm2"`
- `hidden_size = 1024`
- `num_hidden_layers = 16`
- `num_attention_heads = 16`
- `num_key_value_heads = 8`
- `dtype = "bfloat16"`
- `layer_types` 由 `conv` 与 `full_attention` 组成
- sentence-transformers prompt 配置包含：
  - `query: "query: "`
  - `document: "document: "`
- pooling 方式为：
  - `pooling_mode_cls_token = true`
  - 其余 pooling 模式均为 `false`
- 输出维度为 `1024`

因此本次应做如下判断：

1. 该模型属于 **embedding 模型**，不是文本生成模型
2. 该模型是 **双向编码器语义**，不能直接沿用当前 `lfm2.5-350m` 的因果生成前向
3. 该模型当前本地仅确认有 **safetensors** 目录形态
4. 该模型训练依赖 **asymmetric prompts**，省略 `query/document` 前缀会降低检索质量

## 4. 方案选择

### 方案 A：新增独立 `lfm2.5-embedding-350m` family，并走现有 embedding 链路（采用）

优点：

- 与用户选择一致
- 不污染现有 `lfm2.5-350m` 生成语义
- 改动集中在 embedding 路径，回归风险低
- 能完整接入 CLI、服务端、测试与文档

缺点：

- 需要新增一套 family 封装
- 需要为 embedding 增加 prompt 选项透传

结论：采用。

### 方案 B：复用现有 `lfm2.5-350m` family，做双模式扩展

优点：

- 看起来可复用部分底层代码

缺点：

- 当前 `src/models/lfm2_5/model.rs` 明确面向因果生成
- 会把 embedding 语义与生成语义耦合到一个 family 中
- 容易把回归风险扩散到现有 `lfm2.5-350m` 路径

结论：不采用。

### 方案 C：抽象通用 sentence-transformers 适配层

优点：

- 长期可以复用给更多 embedding 模型

缺点：

- 远超本次单模型接入范围
- 当前需求并不要求一次性抽象整类模型

结论：不采用。

## 5. 目标设计

## 5.1 模型标识与元数据

新增 `WhichModel::LFM2_5Embedding350M`，对外表现为：

- CLI / OpenAI model id：`lfm2.5-embedding-350m`
- owner：`LiquidAI`
- model_id：`LiquidAI/LFM2.5-Embedding-350M`
- model_type：`embedding`

并将其加入：

- `LISTED_MODELS`
- `openai_model_id()`
- `owner()`
- `model_id()`
- `model_type()`

保留现有：

- `WhichModel::LFM2_5_350M`
- `openai_model_id() == "lfm2.5-350m"`

两者语义严格分离：

- `lfm2.5-350m`：文本生成
- `lfm2.5-embedding-350m`：embedding

## 5.2 模块布局

新增目录与文件：

- `src/models/lfm2_5_embedding/mod.rs`
- `src/models/lfm2_5_embedding/config.rs`
- `src/models/lfm2_5_embedding/model.rs`
- `src/models/lfm2_5_embedding/generate.rs`
- `src/exec/lfm2_5_embedding.rs`

职责划分：

### `config.rs`

负责读取并校验：

- 基础 `config.json`
- `config_sentence_transformers.json`
- `modules.json`
- `1_Pooling/config.json`

至少暴露：

- 基础 LFM2.5 结构配置
- prompt 映射
- pooling 模式
- normalize 配置
- 输出维度

### `model.rs`

负责：

- safetensors 权重加载
- tokenizer 初始化
- 双向 embedding 前向
- prompt 注入
- CLS pooling
- normalize

### `generate.rs`

负责：

- `init_from_spec(...)`
- `init(...)`
- `embed(...)`
- `embed_with_options(...)`

### `exec/lfm2_5_embedding.rs`

负责 CLI `run` 入口封装，行为与现有 embedding exec 保持一致。

## 5.3 Artifact 设计

本次仅支持：

- `Safetensors`

相关规则：

- `supported_artifacts(LFM2_5Embedding350M) -> [Safetensors]`
- `default_artifact(LFM2_5Embedding350M) -> Safetensors`
- `LoadSpec::validate()` 对该模型只接受 `weight_path`

明确不做：

- GGUF
- ONNX

原因不是“将来永不支持”，而是当前本地目录只验证了 safetensors 形态，本次不伪造额外格式支持。

## 5.4 Prompt 选项设计

新增一个统一 embedding 选项结构，例如：

- `EmbeddingPromptName`
  - `Query`
  - `Document`
- `EmbeddingOptions`
  - `prompt_name: EmbeddingPromptName`

行为规则：

- 默认 `prompt_name = document`
- `prompt_name = query` 时，使用配置中的 `query` prompt
- `prompt_name = document` 时，使用配置中的 `document` prompt
- 如果输入文本本身已经带有 `query: ` 或 `document: ` 前缀，本次不做自动去重或重写

对现有模型的影响：

- `all-minilm-l6-v2`：忽略该选项
- `qwen3-embedding-*`：忽略该选项
- `lfm2.5-embedding-350m`：使用该选项决定前缀

这样可以做到：

1. 对新模型提供必要能力
2. 不改变现有 embedding 模型的行为
3. 不要求调用方自己手写 `query: ` / `document: `

## 5.5 CLI 设计

在 `RunArgs` 中新增：

- `prompt_name: Option<EmbeddingPromptArg>`

建议枚举值：

- `query`
- `document`

行为：

- 仅对 embedding 模型有意义
- 默认值逻辑在内部统一落到 `document`

CLI 使用示例：

```bash
aha run -m lfm2.5-embedding-350m -i "What is the capital of France?" --weight-path D:\model_download\LFM2.5-Embedding-350M --prompt-name query
```

CLI 数据流：

1. `RunArgs.prompt_name`
2. `run_target_model_with_spec(...)`
3. `exec::lfm2_5_embedding::run_with_spec(...)`
4. `Lfm2_5EmbeddingModel::embed_with_options(...)`
5. 输出 embedding JSON

## 5.6 服务端 `/embeddings` 设计

扩展 `EmbeddingRequest`：

- 新增 `prompt_name: Option<String>` 或等价强类型字段

请求示例：

```json
{
  "model": "lfm2.5-embedding-350m",
  "input": "What is the capital of France?",
  "prompt_name": "query"
}
```

行为：

- 未提供时默认 `document`
- 非法值时返回 `400 Bad Request`
- 仍保持 `input` 支持：
  - 单个字符串
  - 字符串数组

服务端数据流：

1. 解析 `input`
2. 解析 `prompt_name`
3. 构造 `EmbeddingOptions`
4. 调用 `guard.instance.embedding(&texts, &options)`
5. 返回 OpenAI 兼容 embedding 响应

## 5.7 `ModelInstance` 与 registry 设计

需要扩展统一分派层：

- `ModelInstance` 增加：
  - `LFM2_5Embedding(...)`
- `ModelInstance::embedding(...)` 改为接收：
  - `input: &[String]`
  - `options: &EmbeddingOptions`

分派逻辑：

- `AllMiniLML6V2` -> 忽略 options
- `Qwen3Embedding` -> 忽略 options
- `LFM2_5Embedding` -> 使用 options

同时更新：

- `src/models/core/registry.rs`
- `src/models/mod.rs`
- 如有 legacy 分支，也要确保新模型不会误走生成路径

该设计的核心目的是：让 prompt 语义成为统一 embedding 接口的一部分，而不是只藏在 CLI 或 API 表层。

## 5.8 模型内部前向设计

本次不直接复用现有 `Lfm2_5Model` 作为 embedding backend。

原因：

- 现有实现使用 causal attention mask
- 现有实现以 `lm_head` 输出 logits
- 现有实现围绕 token-by-token 生成与 cache 设计

而 embedding 模型需要：

- 双向上下文编码
- 输出整段 hidden states
- CLS pooling
- 可选 normalize

因此本次新增独立双向 backend，原则如下：

1. 允许复用 LFM2.5 的底层组件思路与局部辅助逻辑
2. 不把 embedding 语义强行塞入现有生成模型类型
3. 不暴露生成相关 `lm_head`

embedding 前向步骤：

1. 根据 `prompt_name` 取出 prompt 文本
2. 拼接 prompt 与原始文本
3. tokenizer 编码
4. 做双向前向，输出 hidden states
5. 取 CLS 向量
6. 若配置要求，则执行 L2 normalize
7. 返回 `Vec<f32>`

长度策略：

- 第一版沿用模型与 tokenizer 默认限制
- 不新增用户可调的 `max_length` 或其他裁剪参数
- 若需要截断，按现有 tokenizer 行为执行，而不是为该模型单独发明新规则

第一版 batch 策略：

- 逐条编码
- 不做 padding batch 优化

原因：

- 当前项目中其他 embedding family 已采用逐条路径
- 先保证语义正确与接口稳定

## 5.9 配置校验设计

新增 family 在初始化时必须做明确校验：

- `architectures` 包含 `Lfm2BidirectionalModel`
- `modules.json` 至少包含 transformer 模块与 `1_Pooling`
- pooling 模式必须满足：
  - `pooling_mode_cls_token = true`
  - 其他 pooling 模式均为 `false`
- `word_embedding_dimension == hidden_size`
- prompt 配置至少包含：
  - `query`
  - `document`

如果不满足，则初始化直接失败，并在错误中指出缺失或不支持的配置项。

这样可以避免未来用户拿形态相似但头结构不同的目录误加载。

## 5.10 错误处理

需要明确处理以下错误：

1. `weight_path` 缺失
2. 模型目录不存在
3. `config_sentence_transformers.json` 缺失
4. `modules.json` 缺失
5. `1_Pooling/config.json` 缺失
6. `prompt_name` 非法
7. prompt 配置缺少 `query` 或 `document`
8. pooling 模式不是 CLS
9. embedding 输出维度与配置不一致

错误信息原则：

- 明确指出是哪个文件或字段出错
- 不做静默 fallback
- 不把配置错误伪装成通用 “load failed”

## 6. 测试设计

新增测试文件：

- `tests/test_lfm2_5_embedding_safetensors.rs`

覆盖以下场景：

### 6.1 目录与加载

1. 本地目录存在
2. `Lfm2_5EmbeddingModel::init(...)` 可成功加载
3. `LoadSpec` safetensors 校验通过

### 6.2 embedding 结果

1. `init_from_spec(...)` 可输出 embedding
2. 输出条数正确
3. 向量维度为 `1024`

### 6.3 prompt 行为

1. 相同文本在 `query` 与 `document` 下产生不同向量
2. 非法 `prompt_name` 返回错误

### 6.4 基本语义烟测

构造最小检索样例：

- query：`What is the capital of France?`
- relevant document：关于 Paris 的文本
- irrelevant document：无关文本

验证：

- query 向量与相关文档余弦相似度高于与无关文档的相似度

### 6.5 回归点

同步更新：

- `tests/test_load_spec.rs`
  - 新模型只接受 `Safetensors`
- `src/api/mod.rs` 单元测试
  - 新增 `prompt_name` 解析与错误分支测试

## 7. 文档设计

需要更新：

- `README.md`
- `README.zh-CN.md`
- `docs/supported-models.md`
- `docs/supported-models.zh-CN.md`

文档内容至少包含：

1. 模型加入 embedding 分类
2. 本地 `--weight-path` 使用示例
3. `--prompt-name query` 示例
4. `/embeddings` 请求示例

## 8. 非目标

本次明确不做：

1. GGUF 支持
2. ONNX 支持
3. 通用 sentence-transformers 抽象层
4. ColBERT / reranker 共享框架
5. 自动下载与自动目录修复
6. 批量 padding / batching 性能优化
7. 与 Python `sentence-transformers` 的逐值精确对齐验证

## 9. 风险与控制

### 风险 1：错误复用现有 `lfm2_5` 生成实现

影响：

- 可能把 causal 语义带入 embedding 路径
- 容易破坏现有生成模型

控制：

- 使用独立 family
- 不在本次将 `lfm2_5` 改造成双模式

### 风险 2：prompt 逻辑只接到 CLI，没有接到服务端

影响：

- CLI 与 `/embeddings` 行为不一致

控制：

- 把 `EmbeddingOptions` 放进统一 `ModelInstance::embedding(...)` 接口

### 风险 3：未来目录结构略有变化导致误加载

影响：

- 初始化阶段才暴露为难以理解的报错

控制：

- 对 architecture、modules、pooling、prompts 做显式校验

## 10. 预期结果

完成后，用户应能执行：

```bash
aha run -m lfm2.5-embedding-350m -i "Rust embedding test" --weight-path D:\model_download\LFM2.5-Embedding-350M --prompt-name document
```

以及：

```bash
aha run -m lfm2.5-embedding-350m -i "What is the capital of France?" --weight-path D:\model_download\LFM2.5-Embedding-350M --prompt-name query
```

服务端也应能通过 `/embeddings` 正常使用该模型，并在不显式指定时默认按 `document` 编码。
