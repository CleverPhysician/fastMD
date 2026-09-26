# 从三个分支迁移到 fastMD

## 原代码职责

| 来源 | 模型 | 建图 | 捕获实现 | 原调用入口 |
| --- | --- | --- | --- | --- |
| `main` | `matris/model` | `graph/gpu_graph_builder.py`、`RadiusGraph` | `applications/cuda_graph.py` 的容量分桶、静态 workspace、sink padding | `MatRISCalculator`；另有独立 GPU MD |
| `origin/CHGNet_CG` | `chgnet/model` | `graph/gpu_graph_builder.py`、`CrystalGraph` | `model/cuda_graph.py` 的容量分桶、组合能修正 | 原 ASE calculator 不直接接入该 runner；GPU MD 单独使用 |
| `origin/ALIGNN_CG` | `alignn/models` | DGL 和 tensor radius graph | `ff/cuda_graph.py` 的 `ModelCUDAGraphRunner` 和整步 runner | ASE calculator、GPU MD 各有入口 |

三个分支并不是同一包的三个可同时安装的简单插件，而是各自带模型源码的代码树。
具体来源 commit 及文件清单在 [sources.json](sources.json)。

## 新的公共结构

`calculator.py` 只处理 ASE 行为：缓存、属性、单位约定、数组所有权、形状和有限值检查。
`models/base.py` 处理设备选择、能力声明、已知回退、结构签名与捕获失效。
`models/registry.py` 惰性加载模型，非 ALIGNN 用户无需在 import fastmd 时导入 DGL。
各模型适配器负责 checkpoint、原始预测格式、总能量/应力转换、捕获 runner 和统计。

`_vendor` 是当前第一阶段重构保留的推理内核边界，内部仍包含原项目风格的代码。
绝对包导入迁移到 `fastmd._vendor.*`，避免覆盖环境中上游同名包。
只迁移推理相关 Python 和必需数据，没有迁移训练工具、旧样例或平台相关 `.so` 文件。
ALIGNN 的捕获模块与旧 GPU MD 类型位于同一文件，因原模块依赖保留少量 MD 支持文件；
这些不是 fastMD 的公共接口。后续可逐项拆分，不在尚未完成 GPU 数值验证前改写数值内核。

## 本次解决的接口问题

- CHGNet 的模型捕获不再要求用户创建 GPU MD 对象；ALIGNN 使用最小推理上下文驱动 runner。
- CHGNet 推理不再导入无关的 VASP 解析工具，NVML 只在显式请求显存排序时导入。
- 用户用同一个 `FastMDCalculator` 入口，显式能力表防止把 stress/magmoms 宣称为已捕获。
- 默认每次力评估也返回能量，避免 ASE 连续询问 energy/forces 时重复计算。
- 能量输出统一为总能 eV，MatRIS/CHGNet 的 GPa 应力转为 ASE eV/Å³。
- 实现 `free_energy`，适配 ASE force-consistent 优化器。
- 按元素（含顺序）、原子数、cell、PBC 失效捕获，特别处理 CHGNet 捕获时固定的组合能。
- MatRIS 按任务变化重建 runner，避免 EF 捕获被拿去读取应力或磁矩。
- 返回结果做独立复制，避免 replay 修改用户已保存的结果。
- 计算失败先清空 ASE 结果，避免新结构获得旧结果。
- 对当前不支持的非周期/部分周期体系直接报错，防止原 MatRIS calculator 修改用户晶胞。
- 修改私有 MatRIS converter，使显式 CPU 配置在有 GPU 的主机上仍遵守 legacy 建图选择。
- 公开的容量和预热配置从 runner 的环境变量覆盖改为实例参数。
- 捕获缓存有数量上限；未知模型错误或 CUDA 错误向上传播，不静默掩盖。

## 本次范围

本版优先让用户在 ASE 中调用模型级 CUDA Graph。没有统一整步 GPU 积分器，没有增加训练接口，
没有承诺所有 WBM 模型或批处理均已适配。原仓库、分支、权重文件保持原状。

架构参考 [TorchSim ModelInterface](https://github.com/TorchSim/torch-sim/blob/main/torch_sim/models/interface.py)
的统一模型接口及能力声明。fastMD 自己的接口直接接收 ASE `Atoms`，不复制 TorchSim 的状态格式，
也没有复制其实现源码。其批处理、GPU 常驻状态可作为后续独立扩展的参考。
