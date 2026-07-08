# Candle CPU 全局配置优化 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。
>
> **本仓库本地策略：** 直接在当前分支和工作区修改；不要创建分支、worktree 或自动 commit。下面的任务因此不包含任何 `git commit` 步骤。

**目标：** 为 `aha` 的 Candle CPU 路径提供平台感知的 `mkl/accelerate` 默认构建与自动线程配置，同时保持 GPU 路径和现有 safetensors mmap 行为不变。

**架构：** `Cargo.toml` 使用 target-specific dependency 为 `candle-core` 选择 CPU 后端，并增加 `profile.release`。`src/utils/mod.rs` 在第一次落到 `Device::Cpu` 时配置 Rayon 全局线程池；如果用户已经设置 `RAYON_NUM_THREADS`，则完全尊重用户值。由于 Rust 2024 下进程环境变量写入是 `unsafe`，实现不直接调用 `std::env::set_var("RAYON_NUM_THREADS", ...)`，而是保留相同的覆盖语义并安全地使用 `rayon::ThreadPoolBuilder` 配置默认线程数。

**技术栈：** Rust 2024, Candle 0.11, Rayon, sysinfo, Cargo target-specific dependencies

---

## 文件结构

- 修改：`Cargo.toml`
  责任：将 `candle-core` 改为目标平台感知依赖，并增加 `profile.release`。
- 修改：`src/utils/mod.rs`
  责任：为 CPU fallback 增加 Rayon 线程数推断、一次性初始化和对应单元测试。

### 任务 1：为 CPU 路径增加可测试的 Rayon 线程配置辅助函数

**文件：**
- 修改：`src/utils/mod.rs:7-53`
- 修改：`src/utils/mod.rs:1202-1283`
- 测试：`src/utils/mod.rs`

- [ ] **步骤 1：编写失败的测试**

在 `src/utils/mod.rs` 的 `#[cfg(test)] mod tests` 里追加下面三个测试，先锁定“物理核优先、回退到 available parallelism、显式环境变量优先”这三个行为：

```rust
    #[test]
    fn test_resolve_default_rayon_threads_prefers_physical_core_count() {
        assert_eq!(resolve_default_rayon_threads(Some(6), 12), 6);
    }

    #[test]
    fn test_resolve_default_rayon_threads_falls_back_to_available_parallelism() {
        assert_eq!(resolve_default_rayon_threads(None, 12), 12);
        assert_eq!(resolve_default_rayon_threads(Some(0), 12), 12);
        assert_eq!(resolve_default_rayon_threads(None, 0), 1);
    }

    #[test]
    fn test_plan_default_rayon_threads_respects_explicit_env() {
        assert_eq!(
            plan_default_rayon_threads(Some(std::ffi::OsString::from("8")), Some(6), 12),
            None
        );
        assert_eq!(plan_default_rayon_threads(None, Some(6), 12), Some(6));
        assert_eq!(plan_default_rayon_threads(None, None, 12), Some(12));
    }
```

- [ ] **步骤 2：运行测试验证失败**

运行：`cargo test --lib rayon_threads -- --nocapture`

预期：FAIL，报错 `cannot find function 'resolve_default_rayon_threads' in this scope` 和 `cannot find function 'plan_default_rayon_threads' in this scope`。

- [ ] **步骤 3：编写最少实现代码**

在 `src/utils/mod.rs` 顶部补充 `OnceLock`、`OsString` 与 `sysinfo::System` 相关导入，并在 `get_device(...)` 之前新增下列辅助函数。这里先把 CPU 默认配置逻辑做成小而可测的单元，再让 `get_device(...)` 调用它：

```rust
use std::ffi::OsString;
use std::sync::OnceLock;

use sysinfo::System;

static CPU_RAYON_INIT: OnceLock<()> = OnceLock::new();

fn resolve_default_rayon_threads(
    physical_core_count: Option<usize>,
    available_parallelism: usize,
) -> usize {
    match physical_core_count {
        Some(count) if count > 0 => count,
        _ => available_parallelism.max(1),
    }
}

fn plan_default_rayon_threads(
    explicit_env: Option<OsString>,
    physical_core_count: Option<usize>,
    available_parallelism: usize,
) -> Option<usize> {
    if explicit_env.is_some() {
        None
    } else {
        Some(resolve_default_rayon_threads(
            physical_core_count,
            available_parallelism,
        ))
    }
}

fn detect_default_rayon_threads() -> usize {
    let available_parallelism = std::thread::available_parallelism()
        .map(|threads| threads.get())
        .unwrap_or(1);
    let system = System::new_all();
    resolve_default_rayon_threads(system.physical_core_count(), available_parallelism)
}

fn init_cpu_rayon_pool() {
    CPU_RAYON_INIT.get_or_init(|| {
        if let Some(threads) = plan_default_rayon_threads(
            std::env::var_os("RAYON_NUM_THREADS"),
            System::new_all().physical_core_count(),
            std::thread::available_parallelism()
                .map(|threads| threads.get())
                .unwrap_or(1),
        ) {
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global();
        }
    });
}

fn cpu_device() -> Device {
    init_cpu_rayon_pool();
    Device::Cpu
}
```

然后把 `get_device(...)` 中所有返回 CPU 的分支统一改成 `cpu_device()`，尤其是：

```rust
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0).unwrap_or_else(|_| cpu_device())
            }
            #[cfg(all(not(feature = "cuda"), feature = "metal"))]
            {
                Device::new_metal(0).unwrap_or_else(|_| cpu_device())
            }
            #[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
            {
                cpu_device()
            }
```

- [ ] **步骤 4：运行测试验证通过**

运行：`cargo test --lib rayon_threads -- --nocapture`

预期：PASS，3 个新测试全部通过。

### 任务 2：为 Candle CPU 后端增加目标平台默认构建，并验证默认构建不回归

**文件：**
- 修改：`Cargo.toml:9-67`
- 测试：`src/utils/mod.rs`

- [ ] **步骤 1：记录基线构建状态**

运行：`cargo check`

预期：PASS。这一步不是红灯测试，而是配置文件变更前的基线确认；`Cargo.toml` 属于配置文件修改，本任务通过构建校验而不是额外单元测试验证。

- [ ] **步骤 2：修改 `Cargo.toml`**

把通用依赖中的 `candle-core = { version = "0.11.0" }` 移除，保留 `candle-nn`、`candle-transformers` 与现有 feature；随后补上目标平台感知依赖和 `profile.release`：

```toml
[dependencies]
candle-nn = { version = "0.11.0" }
candle-transformers = { version = "0.11.0" }
candle-flash-attn = { version = "0.11.0", optional = true }
serde = "1.0.226"
serde_json = "1.0.145"
anyhow = "1.0.100"
ffmpeg-next = { version = "8.0.0" }
image = "0.25.8"
font8x8 = "0.3.1"
glob = "0.3.3"
reqwest = { version = "0.12.23", features = ["blocking"] }
base64 = "0.22.1"
num = "0.4.3"
minijinja = "2.21.0"
tokenizers = "0.23.1"
aha_openai_dive = { version = "1.4", features = ["stream"] }
uuid = { version = "1.18.1", features = ["v4"] }
chrono = "0.4"
rocket = { version = "0.5.1", features = ["serde_json", "json"] }
tokio = "1.47.1"
hound = "3.5.1"
clap = { version = "4.5.51", features = ["derive"] }
modelscope = "0.1.4"
dirs = "6.0.0"
sysinfo = "0.33"
ctrlc = "3.4.7"
url = "2.5.7"
rayon = "1.1"
realfft = "3.5.0"
symphonia = { version = "0.5.5", features = ["mp3", "wav"] }
serde_yaml = "0.9.34"
zip = "7.2.0"
half = "2.7.1"
byteorder = "1.5.0"
sentencepiece = "0.13.1"
ahash = "0.8.12"
ort = { version = "2.0.0-rc.12", default-features = false, features = ["load-dynamic", "api-24", "half", "ndarray"], optional = true }
ndarray = "0.17"

[target.'cfg(target_os = "macos")'.dependencies]
candle-core = { version = "0.11.0", features = ["accelerate"] }

[target.'cfg(all(not(target_os = "macos"), any(target_arch = "x86", target_arch = "x86_64")))'.dependencies]
candle-core = { version = "0.11.0", features = ["mkl"] }

[target.'cfg(all(not(target_os = "macos"), not(any(target_arch = "x86", target_arch = "x86_64"))))'.dependencies]
candle-core = { version = "0.11.0" }

[profile.release]
lto = true
codegen-units = 1
panic = "abort"
```

保留现有 feature 定义：

```toml
[features]
flash-attn = ["candle-flash-attn"]
cuda = ["candle-nn/cuda", "candle-core/cuda", "candle-transformers/cuda"]
metal = ["candle-nn/metal", "candle-core/metal", "candle-transformers/metal"]
onnx-runtime = ["dep:ort"]
```

- [ ] **步骤 3：运行默认构建验证**

运行：`cargo check`

预期：PASS，说明目标平台依赖声明与现有 feature 定义没有冲突。

- [ ] **步骤 4：运行聚焦回归测试**

运行：`cargo test --lib rayon_threads -- --nocapture`

预期：PASS，前一任务中的 3 个线程配置测试继续通过。

- [ ] **步骤 5：按执行环境补充 feature 验证**

如果当前执行环境是 macOS，运行：`cargo check --features metal`

预期：PASS，`accelerate` 默认 CPU 后端与现有 `metal` feature 可以共存。

如果当前执行环境具备 CUDA 工具链，运行：`cargo check --features cuda`

预期：PASS，`mkl` 默认 CPU 后端不会破坏现有 `cuda` feature 的解析和依赖图。

## 自检

- 规格覆盖度：
  - 平台感知的 `mkl/accelerate` 默认构建：由任务 2 覆盖。
  - CPU 路径自动线程配置且尊重显式环境变量：由任务 1 覆盖。
  - release profile 优化：由任务 2 覆盖。
  - safetensors mmap 保持不变：计划中未触碰模型加载文件，符合规格。
- 占位符扫描：
  - 无 “TODO / 待定 / 后续实现 / 适当处理” 之类占位符。
  - 每个步骤都包含精确文件、代码或命令。
- 类型一致性：
  - 线程配置相关命名统一为 `resolve_default_rayon_threads`、`plan_default_rayon_threads`、`init_cpu_rayon_pool`、`cpu_device`。
  - Cargo 依赖名与现有 feature 引用保持 `candle-core` / `candle-nn` / `candle-transformers` 一致。

## 执行交接

计划已完成并保存到 `docs/superpowers/plans/2026-07-08-candle-cpu-global-config.md`。两种执行方式：

**1. 子代理驱动（推荐）** - 每个任务调度一个新的子代理，任务间进行审查，快速迭代

**2. 内联执行** - 在当前会话中使用 executing-plans 执行任务，批量执行并设有检查点

选哪种方式？
