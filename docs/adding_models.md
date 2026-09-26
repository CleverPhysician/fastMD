# 接入更多模型

fastMD 的稳定边界是 `FastMDCalculator → ModelBackend`。
参考 torch-sim 的能力声明思路，把模型依赖、加载、单位转换、图构建和捕获放在模型适配器里。
不要在 ASE Calculator 中添加 `if model == ...` 分支。

## 最小适配器

以下是可直接运行的接口示例；势能只是测试用的谐振子，不是材料势。

```python
import numpy as np
from ase.build import bulk
from fastmd import FastMDCalculator, ModelBackend, ModelCapabilities, register_model

class HarmonicModel(ModelBackend):
    capabilities = ModelCapabilities(
        properties=frozenset({"energy", "forces"}),
        cuda_graph_properties=frozenset(),
        periodic_only=False,
    )

    def __init__(self, *, checkpoint=None, spring_constant=1.0, **kwargs):
        super().__init__(**kwargs)
        self.spring_constant = spring_constant

    def _predict_eager(self, atoms, properties):
        positions = atoms.positions
        return {
            "energy": 0.5 * self.spring_constant * np.sum(positions**2),
            "forces": -self.spring_constant * positions.copy(),
        }

register_model("harmonic", HarmonicModel)
atoms = bulk("Si", "diamond", a=5.43)
atoms.calc = FastMDCalculator("harmonic", cuda_graph=False,
                              model_kwargs={"spring_constant": 2.0})
print(atoms.get_potential_energy(), atoms.get_forces())
```

也可以惰性注册模块字符串：

```python
register_model("my_wbm_model", "my_package.fastmd_backend:MyModel")
```

构造函数接收 `checkpoint`、`device`、`cuda_graph`，将设备和配置交给 `super().__init__`。
返回的能量是总能量 eV，力是 `(N,3)` eV/Å，应力采用 ASE 符号及 eV/Å³，
可以是 `(3,3)` 或 Voigt `(6,)`。输出必须是独立的 NumPy 数组/标量；不要返回 GPU Tensor，
不要让数组直接引用下一次 replay 会覆盖的 CPU 缓冲。
`predict()` 会统一验证能力、PBC、输入有限性和捕获失效条件。

## 增加 CUDA Graph

1. 在 `capabilities.cuda_graph_properties` 声明真实支持的属性。
2. 实现 `graph_unavailable_reason(properties)`，先调用父类，再检查自身依赖、架构和配置。
   返回 `None` 表示可捕获；返回字符串表示已知不能捕获的原因。不要用宽泛异常掩盖模型错误。
3. 实现 `_predict_graph(atoms, properties)`，将建图与模型捕获区分开。
   GPU 建图不自动意味着能捕获；捕获区不能有 `.item()`、CPU 同步或依赖数据的形状变化。
4. 按模型语义实现 padding/mask、容量增长和捕获输入更新。不同模型的线图和能量归一化不同，
   不能把任意 PyTorch 模型套一层 `torch.cuda.graph` 就认为正确。
5. 实现 `clear_graphs()`，清理静态输入、捕获输出和模型特定的组成缓存，保留加载的权重。
   基类在元素（含顺序）、原子数、晶胞、PBC 改变时调用此方法；位置改变时复用并更新输入。
6. 扩展 `stats()` 报告捕获、重放和容量。不要把“要求使用 graph”当成“已经实际 replay”。

捕获预热需要 `eval()`、冻结权重，并保留坐标/应变求导。不能把依赖能量梯度的推理包在
`torch.no_grad()` 内。模型加载后不要原地改权重、精度或 cutoff；需要改时创建新的后端。

## 接入验证

每个新模型至少用原始实现作为参考验证：

- 两种不同原子数的结构：总能量与每原子能量的转换。
- 非对称扰动结构：力的数值、符号、有限差分，以及应力单位和符号（若支持）。
- 多次 replay：位置/邻居拓扑变化、跨容量阈值、同原子数组分变化。
- 周期条件：非正交晶胞、跨周期边界移动、变胞后重建。
- 旧返回数组保持不变，捕获失败后不留下旧 ASE 结果。
- 对不支持的分子/PBC/孤立原子/模型配置明确报错或标记能力。
- 使用自身权重跑 `examples/compare.py` 和一段短 NVE 轨迹，不仅检查吞吐。

WBM 是评测用的结构集合；数据集名称不替代模型的推理能力声明。
当前注册表不声称已经支持尚未适配的 MACE、SevenNet、ORB 等模型。
未来增加批处理应设计独立的 tensor state / batched backend，不改变 ASE 的单结构调用约定。
