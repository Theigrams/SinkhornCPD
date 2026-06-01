# RPOT
Registration of point clouds on partial optimal transport

## PyTorch 实现 (本项目)

为便于加速与批量实验，本仓库提供了对本目录 MATLAB 代码的 PyTorch 复现：

- 核心实现：`compare_methods/RPOT/rpot_torch.py`
- Python 调用封装：`compare_methods/rpot_wrapper.py`

约定与其它对比方法一致：

```python
from rpot_wrapper import rpot_registration
R, t, info = rpot_registration(source, target)
# 对齐后的点：source_aligned = source @ R.T + t
```

## 结果一致性验证 (Torch vs MATLAB)

本目录自带 `data.mat` 示例数据。你可以用下面脚本做严格数值对比：

```bash
python compare_methods/RPOT/verify_torch_vs_matlab.py
```

它会：

1) 调用 MATLAB 运行本目录原始实现，导出 `R0, t0`；  
2) 运行 PyTorch 实现；  
3) 打印 `max_abs_err` 等误差统计（旋转/平移/对齐后点云）。
