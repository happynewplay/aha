# Add "Using Local Model Weights" README Section

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在 README.md 中新增一个结构化的"使用本地模型权重"章节，让读者快速了解如何跳过自动下载、直接加载本地已有的模型权重。

**架构：** 纯文档变更，在 README.md 的 "CLI Quick Reference" 代码块之后、"Chat" 章节之前插入一个 `### Using Local Model Weights` 新章节。内容包含参数参考表格、三个按格式分类的子节（Safetensors / GGUF / ONNX），以及一个注意事项备注。

**技术栈：** Markdown

---

## 文件结构

| 文件 | 操作 | 职责 |
|------|------|------|
| `README.md` | 修改 | 在 "CLI Quick Reference" 之后插入新章节 |

## 任务 1：编辑 README.md 插入新章节

**文件：**
- 修改：`README.md`

### 步骤 1：在 "CLI Quick Reference" 代码块结束后、"### Chat" 之前，插入以下新内容

当前 `README.md` 第 119 行是关闭反引号 `` ``` ``（CLI Quick Reference 代码块结束），第 121 行是空行，第 123 行是 `### Chat`。

在第 120 行（`CLI Quick Reference` 代码块结束后的空行）之后、`### Chat` 之前，插入以下内容：

```markdown
### Using Local Model Weights

If you already have model weights downloaded on disk, you can skip the automatic download and load them directly using the path options below.

#### Parameters

| Parameter | Description | Required For |
|-----------|-------------|--------------|
| `--weight-path <path>` | Directory containing safetensors weights | Safetensors (default) |
| `--artifact-format {auto,safetensors,gguf,onnx}` | Model artifact format | GGUF / ONNX |
| `--gguf-path <path>` | Path to a GGUF model file or directory | GGUF |
| `--onnx-path <path>` | Path to an ONNX model file or directory | ONNX |
| `--tokenizer-dir <path>` | Directory containing tokenizer config files | GGUF / ONNX |
| `--mmproj-path <path>` | Path to multimodal project GGUF weights | Multimodal GGUF |

#### Safetensors (Default)

Safetensors is the default artifact format. Simply provide the directory containing the weight files via `--weight-path`:

```bash
# Embedding model
aha run -m all-minilm-l6-v2 -i "Rust embedding test" --weight-path D:\models\all-MiniLM-L6-v2

# Text model
aha run -m qwen3-0.6b -i "Hello" --weight-path D:\models\qwen3-0.6b
```

#### GGUF

For GGUF weights, set `--artifact-format gguf` and point `--gguf-path` at the `.gguf` file or directory containing it:

```bash
# Text generation with quantized GGUF
aha run -m qwen3-0.6b -i "Hello" --artifact-format gguf --gguf-path D:\models\qwen3-0.6b-Q4_K_M.gguf --tokenizer-dir D:\models\tokenizer

# Multimodal with MMProj (vision + text)
aha run -m qwen3-0.6b -i "Describe this image" --artifact-format gguf --gguf-path D:\models\qwen3-0.6b-Q4_K_M.gguf --tokenizer-dir D:\models\tokenizer --mmproj-path D:\models\mmproj.gguf

# OCR
aha run -m glm-ocr -i ocr.png --artifact-format gguf --gguf-path D:\models\GLM-OCR-GGUF
```

#### ONNX

For ONNX models, set `--artifact-format onnx` and provide the ONNX directory/file via `--onnx-path`:

```bash
# Embedding model
aha run -m all-minilm-l6-v2 -i "Rust embedding test" --artifact-format onnx --onnx-path D:\models\onnx --tokenizer-dir D:\models\tokenizer

# OCR
aha run -m glm-ocr -i ocr.png --artifact-format onnx --onnx-path D:\models\GLM-OCR-ONNX --tokenizer-dir D:\models\GLM-OCR-ONNX
```

> **Note:** When using GGUF or ONNX formats, `--tokenizer-dir` is required to load the tokenizer configuration. When using multimodal GGUF models (e.g. Qwen3-VL), `--mmproj-path` is also required.
```

**关键注意事项：**
- 插入位置是第 120 行（`CLI Quick Reference` 代码块结束后的空行）之后
- 新章节的标题是 `### Using Local Model Weights`（三级标题，与 "Chat"、"Supported Models" 等平级）
- 每个子节（Safetensors / GGUF / ONNX）的标题是 `####`（四级标题）
- 示例中的路径使用 `D:\models\...` 作为通用占位风格，与 README 现有风格一致
- 不要修改 "CLI Quick Reference" 代码块中的已有命令（那些仍然保留）

- [ ] **步骤 1：在 README.md 中插入新内容**

将上述 Markdown 内容插入到 `README.md` 第 120 行之后（即 "CLI Quick Reference" 代码块结束后的空行之后、`### Chat` 之前）。

- [ ] **步骤 2：验证文件结构正确**

用 Read 工具确认插入后文件结构如下：
- 第 1-119 行：原有内容不变（到 "CLI Quick Reference" 代码块结束）
- 第 120 行起：`### Using Local Model Weights` 新章节
- 新章节之后：原有的 `### Chat` 及后续内容保持不变

- [ ] **步骤 3：Commit**

```bash
git add README.md
git commit -m "docs: add 'Using Local Model Weights' section to README"
```
