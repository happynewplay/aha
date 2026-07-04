# MiniCPM5 Request-JSON CLI 设计说明

日期：2026-07-04  
主题：为 `aha run --model minicpm5-1b` 增加 `--request-json`，使 CLI 能传入完整 `ChatCompletionParameters` 并支持工具调用请求格式

## 1. 背景与问题

当前 `src/exec/minicpm5.rs` 只接受一段文本输入，并在内部固定构造：

- 单条 `user` message
- 无 `tools`
- 无 `tool_choice`
- 无多轮 `messages`

因此即使 MiniCPM5 模型层已经具备：

- chat template 渲染 `tools`
- 非流式 `tool_calls` 响应封装
- 流式 `<tool_call>...</tool_call>` 片段解析

CLI `run` 仍然无法把完整请求传进去，导致用户无法从命令行验证 MiniCPM5 的工具调用输出链路。

## 2. 本次目标

本次仅做最小改造：

1. 为 `run` 子命令新增 `--request-json <file>` 参数。
2. 当该参数存在时，从文件读取完整 `ChatCompletionParameters`。
3. `MiniCPM5Exec` 优先使用该完整请求，不再强制降级成单条文本消息。
4. 保持现有普通文本输入兼容。
5. 为该行为补测试。

## 3. 非目标

本次明确不做：

1. 自动执行 `tool_calls`
2. 将 tool result 追加为 `role=tool` 后继续多轮推理
3. 为所有 exec 一次性统一接入 `--request-json`
4. 新增 `--tools-json`、`--tool-choice` 等细粒度参数
5. 改动服务端 `/chat` 路径

也就是说，本次结果是：

- CLI 可以把带 `tools` 的请求送进 MiniCPM5
- 模型如果产出 `tool_calls`，CLI 会正常打印该响应
- 但 CLI 不会执行任何工具

## 4. 方案选择

### 方案 A：新增 `--request-json`（采用）

优点：

- 与 `ChatCompletionParameters` 完全对齐
- 一次性支持 `messages`、`tools`、`tool_choice`、`metadata`、`max_tokens`
- 改动面最小，不需要扩展大量 CLI 参数
- 便于后续推广到其他文本模型

缺点：

- 用户需要准备一个 JSON 文件

结论：采用。

### 方案 B：新增 `--tools-json` 等零散参数

优点：

- 单次调用看起来更直接

缺点：

- 参数很快膨胀
- 多轮消息支持仍然不自然
- 需要做较多拼装逻辑

结论：不采用。

### 方案 C：直接实现完整 tool loop

优点：

- 一步到位

缺点：

- 远超 `minicpm5.rs` 的最小修复范围
- 涉及工具注册、执行权限、多轮回填、错误策略

结论：不采用。

## 5. 设计

## 5.1 CLI 参数

在 `src/main.rs` 的 `RunArgs` 中新增：

- `request_json: Option<String>`

行为：

- `--request-json <file>` 指向一个本地 JSON 文件
- 文件内容必须能反序列化为 `aha_openai_dive::v1::resources::chat::ChatCompletionParameters`

约束：

- 本次不新增 stdin 模式
- 本次不支持直接把 JSON 字符串放进命令行参数

## 5.2 MiniCPM5Exec 请求解析

在 `src/exec/minicpm5.rs` 中，将 `run_with_spec` 的请求构造拆成两条路径：

### 路径 1：`request_json` 存在

执行流程：

1. 读取 JSON 文件
2. 反序列化为 `ChatCompletionParameters`
3. 直接传给 `model.generate(...)`

该路径下：

- 不再读取 `input[0]` 作为必填文本
- `tools`、`tool_choice`、多轮 `messages` 完整保留

### 路径 2：`request_json` 不存在

保持现有兼容行为：

1. 从 `input.first()` 读取文本或 `file://` 文本文件
2. 构造单轮 `user` message
3. 传给 `model.generate(...)`

这样现有调用方式不被破坏。

## 5.3 函数拆分

为了让行为可测，`src/exec/minicpm5.rs` 中建议新增小函数：

- `build_text_request(input: &[String]) -> Result<ChatCompletionParameters>`
- `load_request_json(path: &str) -> Result<ChatCompletionParameters>`
- `build_request(input: &[String], request_json: Option<&str>) -> Result<ChatCompletionParameters>`

目标：

- 将“读取文本文件”和“读取 JSON 请求文件”分开测试
- 让 `run_with_spec` 保持薄封装

## 5.4 `run_target_model_with_spec` 透传

当前 `run_target_model_with_spec` 对 MiniCPM5 的调用是：

- `MiniCPM5Exec::run_with_spec(&args.input, args.output.as_deref(), spec)?;`

本次需要把 `args.request_json.as_deref()` 也一并传下去。

因此 `MiniCPM5Exec::run_with_spec` 需要改签名，例如：

- `run_with_spec(input, output, spec, request_json)`

`ExecModel::run(...)` trait 保持不变；其默认分支传 `None` 即可。

## 5.5 输出行为

本次不改变 CLI 的整体输出语义，但建议顺手把最终响应序列化成 JSON 输出，而不是 `Debug` 字符串。

原因：

- `tool_calls` 是结构化字段
- `serde_json::to_string_pretty(&result)` 更适合作为 CLI 结果

如果改动过大，则至少保证 `request-json` 路径输出结果中能稳定看到 `tool_calls`。

本次推荐直接统一改为 JSON 输出。

## 5.6 错误处理

需要明确处理以下错误：

1. `--request-json` 指向的文件不存在
2. 文件存在但不是合法 JSON
3. JSON 合法但不符合 `ChatCompletionParameters` 结构
4. 未提供 `--request-json` 时，`input` 为空
5. 文本输入是 `file://`，但目标文件不可读

错误信息原则：

- 指出是哪个路径出错
- 指出是“读取失败”还是“反序列化失败”
- 不做静默 fallback

特别是：

- 如果用户显式传了 `--request-json`，即使 `input` 为空也不应报 “requires one text input”

## 6. 测试设计

本次遵循最小 TDD，先补失败测试，再实现。

## 6.1 `src/exec/minicpm5.rs` 单元测试

增加对纯请求构造函数的测试：

1. `build_request` 在给定 `request_json` 时，优先读取 JSON 文件
2. JSON 请求中包含 `tools` 时，结果对象保留 `tools`
3. 未提供 `request_json` 时，仍按旧逻辑构造单条 `user` message
4. 文本 `file://` 输入仍然可读
5. `request_json` 非法时返回错误

这些测试不依赖真实模型权重。

## 6.2 `src/main.rs` CLI 解析测试

补一个参数解析测试，确认：

- `run --model minicpm5-1b --request-json req.json`

能把 `request_json` 正确解析到 `RunArgs`。

## 6.3 可选集成测试

如果本地测试基建方便，可补一个轻量集成测试：

- 传入一个带 `tools` 的 request JSON
- 断言请求已正确传给模型初始化前的构造层

但本次不要求真实跑 MiniCPM5 权重。

## 7. 影响面

直接影响文件：

- `src/main.rs`
- `src/exec/minicpm5.rs`

可能新增：

- `tests/...` 或在 `src/exec/minicpm5.rs` 内补测试模块

不应影响：

- `src/models/minicpm5/generate.rs`
- 服务端 `/chat`
- 其他 exec 模块

## 8. 成功标准

满足以下条件即认为完成：

1. `run` 子命令能解析 `--request-json`
2. `MiniCPM5Exec` 能从 JSON 文件读取完整 `ChatCompletionParameters`
3. 带 `tools` 的请求能送到 MiniCPM5 模型层
4. 现有文本输入模式不回归
5. 新增测试先失败、后通过

## 9. 最终结论

本次采用最小方案：

- 仅为 MiniCPM5 CLI `run` 增加 `--request-json`
- 用完整请求对象打通 tool-calling 请求输入链路
- 不实现工具自动执行

这样可以最低风险地验证：`MiniCPM5` 在当前仓库中是否已经具备“接收 tools 并返回 `tool_calls`”的 CLI 能力。
