# WBM 模型后端

fastMD 现在可以通过同一个 `FastMDCalculator` 接口加载 WBM 的五类模型：

```python
from fastmd import FastMDCalculator

calc = FastMDCalculator("dpa4", checkpoint="DPA4.pt", device="cuda")
# 可选名称：nequip、orbv3、sevennet、tace
```

这五个后端都返回 ASE 所需的总能量和原子力，并保留模型包自己的单位、邻居表和权重加载逻辑。`cuda_graph=False` 使用 eager 路线；DPA4、NequIP、ORB-v3 和 SevenNet 在 CUDA 上默认选择已经发布的 model-only CUDA Graph 路线。TACE 当前只接入单结构 eager 后端；它原有的固定邻居和整步弛豫捕获属于 relaxation/session 层，不能冒充成这个单结构接口的 model-only graph。

WBM 模型包不是 fastMD 的 PyPI 依赖，需要在目标环境中单独安装，并确保其 `md_stages.opt1/opt2` 接口可导入。fastMD 只在实际选择对应模型时导入这些包，因此不影响已有的 MatRIS、CHGNet 和 ALIGNN 用户。

当前接口仍然一次处理一个 ASE `Atoms`。固定晶胞的 opt3 GPU FIRE 核心位于
`fastmd.relaxation`，通过后端的 `device_callback` 接口调用，避免每一步经过
NumPy。它只接受固定形状、无约束结构；动态邻居容量恢复、变胞和多结构批处理仍
需要各模型单独实现，不能从 `predict` 接口反推成全 GPU 路径。本次没有把旧的
实验性批处理脚本复制进公共 calculator。
