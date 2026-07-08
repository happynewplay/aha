# Candle CPU 全局配置优化设计说明

日期：2026-07-08  
主题：参考 `推理优化.md` 优化 `aha` 的全局 Candle CPU 配置

## 1. 背景与目标

当前仓库中的 Candle 相关配置分散在两层：

- 构建层：`Cargo.toml` 仅依赖基础 `candle-core` / `candle-nn` / `candle-transformers`
- 运行时层：`src/utils/mod.rs` 中统一处理 `Device` 与 `DType`，各模型 `init(...)` 会复用该入口

已有现状：

1. safetensors 加载路径已经大量使用 `VarBuilder::from_mmaped_safetensors(...)`
2. CPU 路径下没有统一的线程数初始化
3. Candle CPU 后端未默认启用 `mkl` 或 `accelerate`
4. `profile.release` 没有针对 CPU 推理做额外优化

参考 `推理优化.md`，本次目标是将“对 CPU 推理普遍有益、且能稳定落地的优化”收敛为项目级默认行为，而不是要求每个调用者单独记忆构建参数或环境变量。

本次目标：

1. 为 Candle 增加平台感知的 CPU 加速后端默认构建策略
2. 在 CPU 路径统一初始化 Rayon 线程数
3. 保持 safetensors mmap 加载路径不变
4. 为 release 构建补充适合 CPU 推理的 profile 配置
5. 保证用户手动设置的线程环境变量优先
6. 不引入对当前 GPU 路径的行为回归

## 2. 已确认约束

### 用户确认

- 方向选择：方案 A
  - 代码内全局配置
  - `Cargo.toml` 中按平台默认启用 CPU 加速特性
- 线程策略：自动推断 + 可覆盖
  - 仅在代码中自动设默认值
  - 若用户已设置环境变量，则完全尊重用户值
- CPU 后端特性策略：
  - 如果默认开启不会伤害不支持的目标平台，就作为默认构建启用
  - 但不能为了“默认开启”牺牲跨平台构建稳定性

### 技术事实

基于仓库和 Candle 文档，可确认：

1. `candle-core` 提供可选 CPU 后端：
   - `mkl`
   - `accelerate`
2. `accelerate` 面向 Apple Accelerate，属于 macOS 路径
3. `mkl` 面向 Intel MKL，适合作为 x86 / x86_64 CPU 路径的可选后端
4. 当前仓库已经通过 `VarBuilder::from_mmaped_safetensors(...)` 使用 mmap safetensors 加载
5. `src/utils/mod.rs::get_device(...)` 是当前最合适的“全局 Candle 运行时入口”

### 明确非目标

本次不做：

1. 在仓库中硬编码 `RUSTFLAGS="-C target-cpu=native"`
2. 自动做 CPU core affinity 绑核
3. 新增 CLI 参数让用户传线程数
4. 修改各模型内部前向逻辑、KV cache 或 attention 实现
5. 针对 decode / prefill 分别设置不同线程数
6. 调整 safetensors 加载方式

## 3. 问题定义

当前 CPU 路径存在三个实际问题：

### 3.1 CPU 后端未默认优化

`Cargo.toml` 中当前 Candle 依赖为：

- `candle-core = { version = "0.11.0" }`
- `candle-nn = { version = "0.11.0" }`
- `candle-transformers = { version = "0.11.0" }`

这意味着：

1. x86 / x86_64 主机默认不会走 MKL
2. macOS 默认不会走 Accelerate
3. 用户即使阅读过 `推理优化.md`，也必须自己记住额外 feature

### 3.2 CPU 线程数没有统一初始化

当前仓库没有设置 `RAYON_NUM_THREADS`，效果是：

1. Rayon 会按自身默认值运行
2. 在异构 CPU 或超线程场景，默认线程数可能不是最合适选择
3. 用户只有通过外部环境变量才能控制线程数

### 3.3 release profile 仍是默认配置

当前 `Cargo.toml` 没有显式 `[profile.release]` 配置，因此：

1. 没有开启 `lto = true`
2. 没有将 `codegen-units` 收敛到更利于最终优化的配置
3. 没有 `panic = "abort"` 这类有利于减小二进制与减少部分开销的配置

## 4. 方案比较

### 方案 A：平台感知的默认 CPU 后端 + CPU 线程自动初始化 + release profile 优化（采用）

设计：

1. 在 `Cargo.toml` 中按目标平台配置 Candle CPU 后端
2. 在 `src/utils/mod.rs` 中增加一次性的 CPU 运行时初始化
3. 仅在最终设备为 `Device::Cpu` 且用户未手动设置时自动设置 `RAYON_NUM_THREADS`
4. 增加适合 CPU 推理的 `profile.release`

优点：

1. 与 `推理优化.md` 的高收益项一致
2. 对调用者透明，默认构建和默认运行就能受益
3. 不要求用户记忆额外 feature 或环境变量
4. 兼容用户手动覆盖
5. 改动集中在全局入口，不污染各模型实现

缺点：

1. 需要谨慎处理不同目标平台的依赖声明
2. 需要新增线程初始化测试

结论：采用。

### 方案 B：只做运行时线程策略和 release profile，不动 Candle CPU 后端

优点：

1. 兼容性风险最低
2. 改动最小

缺点：

1. 错过 `mkl` / `accelerate` 这一类最直接的 CPU 后端收益
2. 与 `推理优化.md` 的主诉求不完全一致

结论：不采用。

### 方案 C：方案 A 再加仓库级 `target-cpu=native`

优点：

1. 在单机部署场景下可能进一步提速

缺点：

1. 构建产物和构建机强绑定
2. 会影响交叉构建和可移植性
3. 不适合做成仓库默认行为

结论：本次不采用。

## 5. 总体设计

本次采用“构建层 + 运行时层”的双层优化。

### 5.1 构建层

在 `Cargo.toml` 中引入目标平台感知的 Candle CPU 后端：

- macOS：默认启用 `accelerate`
- x86 / x86_64 的非 macOS 目标：默认启用 `mkl`
- 其他目标：不启用这两个 CPU 后端

设计原则：

1. 不是“所有平台无条件启用”
2. 而是“对对应目标平台默认启用正确后端”
3. 避免因为目标平台不适配而造成构建失败

### 5.2 运行时层

在 `src/utils/mod.rs` 中新增 CPU 推理运行时初始化逻辑，职责：

1. 仅在代码第一次走到 CPU 默认设备路径时执行一次
2. 检查 `RAYON_NUM_THREADS` 是否已由用户设置
3. 若未设置，则自动推断一个默认线程数并写入环境变量

### 5.3 Release 层

在 `Cargo.toml` 中新增：

```toml
[profile.release]
lto = true
codegen-units = 1
panic = "abort"
```

理由：

1. 这些配置与 CPU 推理二进制优化方向一致
2. 不依赖具体模型 family
3. 对当前项目的 CLI / service 二进制都有效

## 6. 构建配置设计

## 6.1 平台感知的 Candle 依赖

推荐将 Candle 依赖拆为目标平台分组，而不是继续只保留一组无 feature 的通用依赖。

目标：

### macOS

- `candle-core` 启用 `accelerate`

### Windows / Linux 的 x86 / x86_64

- `candle-core` 启用 `mkl`

### 其他目标

- `candle-core` 保持无 CPU 加速 feature

`candle-nn` 与 `candle-transformers` 继续维持当前版本和职责，不需要单独加 `mkl` / `accelerate` feature。

关键点：

1. `mkl` / `accelerate` 只在 `candle-core` 上声明
2. 依赖声明必须避免目标平台冲突
3. 现有 `cuda` / `metal` feature 语义保持不变

## 6.2 对现有 feature 的影响

需要保持以下语义不变：

- `flash-attn`
- `cuda`
- `metal`
- `onnx-runtime`

本次 CPU 后端优化不应改变：

1. `--features cuda` 的可用性
2. `--features metal` 的可用性
3. ONNX 路径的可用性

也就是说：

1. CPU 后端默认优化与 GPU feature 共存
2. 设备选择逻辑仍由现有 `get_device(...)` 决定
3. 本次不新增新的顶层 Cargo feature 开关

## 7. 运行时线程策略设计

## 7.1 触发条件

线程自动配置必须满足全部条件才生效：

1. 调用方没有显式传入 `device`
2. `get_device(...)` 最终解析为 `Device::Cpu`
3. 进程环境中不存在 `RAYON_NUM_THREADS`
4. 当前进程尚未执行过这段 CPU 初始化

若任何条件不满足，则不做环境变量写入。

## 7.2 自动推断规则

默认线程数不使用“逻辑核数”直接硬上，而采用“物理核数优先”的策略。

推荐实现：

1. 使用 `std::thread::available_parallelism()` 获取并行度上界
2. 使用 `sysinfo` 获取 CPU 物理核心数
3. 当能拿到物理核心数且大于 0 时，使用物理核心数
4. 否则回退到 `available_parallelism()`
5. 最低保证为 `1`

理由：

1. 与 `推理优化.md` 中“CPU 推理优先考虑物理核数”的建议一致
2. 比硬编码 `6` 或“逻辑核数的一半”更通用
3. 在未知或受限环境中仍有回退路径

## 7.3 与用户配置的关系

若用户已经设置了：

- `RAYON_NUM_THREADS`

则代码完全不覆盖，不打印警告，不做二次修正。

这是本次设计的硬约束，因为用户可能：

1. 已做过 benchmark
2. 在异构 CPU 上有自己的绑核策略
3. 在容器环境中按 quota 手动限线程

## 7.4 初始化位置

推荐在 `src/utils/mod.rs` 新增类似：

- `init_cpu_inference_env()`
- `resolve_default_rayon_threads()`

并由 `get_device(None)` 在落到 CPU 分支时触发。

不放到 `main.rs` 的原因：

1. 当前项目既有 CLI / serv，也有 tests 和直接模型初始化路径
2. 许多模型 `init(...)` 会直接复用 `get_device(...)`
3. 将逻辑放在 `utils` 更接近真实全局入口

不放到每个模型 `init(...)` 的原因：

1. 重复
2. 易遗漏
3. 会让“全局配置”退化成“各模型各自处理”

## 7.5 执行次数

该初始化必须是一次性的。

推荐用：

- `OnceLock<()>`
- 或 `Once`

保证：

1. 并发初始化时不会重复写环境变量
2. 多模型加载时只执行一次
3. tests 中也能稳定复用

## 8. safetensors mmap 路径设计

本次明确保持现状，不改动以下模式：

- `VarBuilder::from_mmaped_safetensors(...)`

理由：

1. 当前仓库已广泛采用该路径
2. `推理优化.md` 也明确将 mmap 视为正确方向
3. 本次目标是全局配置优化，而不是模型加载重写

因此本次不会：

1. 替换为全量读入内存
2. 改造为新的权重缓存抽象
3. 修改各模型的权重枚举逻辑

## 9. 错误处理设计

### 9.1 线程初始化失败

如果线程数推断过程中某个信息源不可用：

1. 不应使模型初始化失败
2. 应回退到可用的并行度值
3. 若仍无法推断，则回退到 `1`

线程初始化是优化项，不是正确性前提。

### 9.2 环境变量写入

环境变量写入属于进程内操作。

若出现异常路径：

1. 不抛出对外错误
2. 以“保守回退，不阻塞加载”为原则

### 9.3 平台依赖配置

若某平台不属于：

- macOS
- x86 / x86_64 的 Windows / Linux

则不启用 `mkl` / `accelerate`，避免目标平台构建时落入不支持依赖。

## 10. 测试设计

本次采用先补测试再实现的方式。

## 10.1 单元测试

优先在 `src/utils/mod.rs` 或对应测试文件中覆盖：

1. 当 `RAYON_NUM_THREADS` 已存在时，初始化不覆盖用户值
2. 当 `RAYON_NUM_THREADS` 不存在时，初始化会写入一个正整数
3. 推断线程数的辅助函数返回值至少为 `1`

为保证测试可控，建议将“实际写环境变量”和“线程数推断”拆开：

- 纯函数：负责解析默认线程数
- 带副作用函数：负责只在需要时设置环境变量

这样可以避免测试严重依赖全局进程状态。

## 10.2 构建验证

需要至少验证：

1. 默认 `cargo test` 或 `cargo check` 仍能在当前开发机通过
2. `--features cuda`
3. `--features metal`

本次不要求在当前会话中真正跨平台执行所有构建，但设计上必须保证目标平台声明不冲突。

## 10.3 回归风险检查

重点关注：

1. `get_device(Some(...))` 路径行为不变
2. 显式 GPU 设备路径不会被 CPU 初始化污染
3. 已有模型初始化测试不因线程策略而失效

## 11. 实施顺序建议

1. 在 `utils` 层新增线程推断与一次性初始化的失败测试
2. 修改 `Cargo.toml`，加入目标平台感知的 Candle CPU 后端依赖
3. 增加 `[profile.release]` 配置
4. 在 `src/utils/mod.rs` 实现 CPU 初始化逻辑
5. 运行相关测试与构建验证
6. 如有必要，补充简短文档说明默认行为

## 12. 风险与控制

### 风险 1：目标平台依赖写错导致构建冲突

控制：

1. 使用目标平台感知依赖声明
2. 不用单一依赖同时混入 `mkl` 和 `accelerate`

### 风险 2：测试因进程级环境变量互相污染

控制：

1. 尽量将推断逻辑做成纯函数
2. 将“是否设置环境变量”的逻辑与“计算线程数”解耦
3. 在测试中显式清理和恢复环境变量

### 风险 3：GPU 路径被误触发 CPU 初始化

控制：

1. 只在最终分支进入 `Device::Cpu` 时初始化
2. `Some(device)` 分支保持原样返回，不额外做推理环境写入

## 13. 最终设计结论

本次采用以下最终设计：

1. 在 `Cargo.toml` 中按目标平台默认启用 Candle CPU 后端：
   - macOS -> `accelerate`
   - x86 / x86_64 的非 macOS 目标 -> `mkl`
   - 其他目标 -> 不启用 CPU 加速后端
2. 在 `src/utils/mod.rs` 增加一次性的 CPU 运行时初始化
3. 仅当最终设备为 CPU 且用户未设置 `RAYON_NUM_THREADS` 时，自动将其设为推断出的默认线程数
4. 默认线程数采用“物理核心数优先，无法获取时回退到可用并行度，最低为 1”的规则
5. 保持现有 `from_mmaped_safetensors(...)` 路径不变
6. 增加适合 CPU 推理的 `profile.release`
7. 不将 `target-cpu=native`、绑核或模型内部重写纳入本次范围

该设计能将 `推理优化.md` 中最稳定、最通用的收益点收敛到仓库默认行为，同时避免把构建产物与单机硬件强绑定，也不会要求调用方手动记忆额外配置。
