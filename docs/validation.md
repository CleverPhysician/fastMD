# 本次验证记录

日期：2026-09-26。

## 环境与权重

- Python 3.11，PyTorch 2.8.0+cu128，ASE 3.23.0。
- ALIGNN 检查使用现有 `alignn` 环境，DGL 1.1.1+cu118。
- 当前节点 `torch.cuda.is_available() == False`，没有可用 NVIDIA 驱动。
- CHGNet：随包的 0.3.0 权重。
- MatRIS：工作区 `checkpoint/MatRIS_10M_OAM.pth.tar`。
- ALIGNN：从 `origin/ALIGNN_CG` 提取的 `v12.2.2024_dft_3d_307k`，仅复制到临时目录供测试。
- 未修改现有 conda 环境；测试工具使用临时目录中的纯 Python pytest 依赖。

## 已完成

完整测试结果：**29 passed, 3 skipped**。跳过的 3 项分别是三个模型的真实 CUDA replay 测试。

测试包含：

- ASE 能量、力、stress Voigt 排列、`free_energy` 和结果缓存。
- warmup 不改变结构，ASE 约束与 VelocityVerlet 可以运行。
- 位置变化重新推理，元素/顺序/原子数/晶胞变化触发捕获失效。
- 错误输入不保留旧结果；严格模式不静默回退；自动模式报告回退原因。
- 三个真实模型的 CPU 推理、输出形状、有限值和力的有限差分检查。
- MatRIS、CHGNet 真实模型的磁矩和应力。
- MatRIS、CHGNet 用于捕获的 sink padding 在 CPU 上与非填充预测一致。
- ALIGNN 静态 mask 前向函数在 CPU 上与 DGL 动态模型一致。
- ALIGNN 默认总能量/力与原分支 Calculator 输出一致。

ALIGNN 的默认 Calculator 会将力乘以训练 `batch_size`。本地模型该值为 6，
因此其有限差分检查针对除去该因子后的模型力；没有为了通过测试而改变默认推理结果。
这也意味着默认 force scaling 下不能假定返回力等于返回总能量的负梯度。

额外完成：

- Python 源码及示例编译检查。
- 构建 wheel，检查包含 CHGNet 权重和三个来源许可证，没有依赖旧仓库路径。
- 将 wheel 安装到独立临时目录，从 `/tmp` 成功运行 CHGNet 单点推理。
- 确认 `import fastmd` 不导入 DGL/pymatgen；模型运行不注册顶层 `chgnet`、`matris`、`alignn` 包。
- 使用安装后的包运行 ASE Langevin 示例 2 步，生成轨迹和日志。
- CHGNet 推理额外检查不依赖 NVML 和 VASP 解析所需的 h5py。
- 原 `MatRIS-09bk` 的 `git status --short` 保持为空。

## 尚未在本节点验证

CUDA 捕获/重放、GPU 邻居构建、融合 Triton 内核、多 GPU 设备行为、显存占用和加速比。
CPU 上的填充/掩码一致性只验证数值表达式，不能替代这些 GPU 验证。
本次没有给出未经测量的性能提升数字。

GPU 节点验收：

```bash
python -m pip install -e '.[matris,chgnet,cuda,test]'
# ALIGNN 需另行准备匹配的 DGL 环境，再安装 alignn extra
export FASTMD_MATRIS_CHECKPOINT=/absolute/path/MatRIS_10M_OAM.pth.tar
export FASTMD_ALIGNN_CHECKPOINT=/absolute/path/alignn_model_directory
python -m pytest -q -m cuda
python examples/compare.py --model chgnet
python examples/compare.py --model matris --checkpoint "$FASTMD_MATRIS_CHECKPOINT"
python examples/compare.py --model alignn --checkpoint "$FASTMD_ALIGNN_CHECKPOINT"
```

`cuda_graph=True` 会要求实际捕获；缺依赖或不支持属性时测试应失败，而不是将 eager 当作 GPU 通过。
另外应针对实际生产结构测试容量增长、非正交晶胞、周期边界和短 NVE 守恒。
