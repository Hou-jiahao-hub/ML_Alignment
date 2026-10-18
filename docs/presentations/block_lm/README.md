# Sensor block-LM 展示图

## 草图审视

原草图正确表达了固定 helix、移动 target sensor 和随之改变的交点。
需要补全的是固定局部测量、面内残差，以及一个 sensor 汇总多条径迹。
不能用 nominal 黑点与候选白点的全局距离代替求解目标。

当前实现计算 `measured_local - Rᵀ(intersection-center)[:2]`，对有效
LOC 分量按测量标准差归一化，逐 sensor 构建几何 Jacobian 和 LM 更新。
候选在本次访问固定的分量集合上降低总 chi2、保留全部原有效分量并通过
失效率检查才接受。集合在下次线性化时可扩大。

## 交付

- `data/presentations/block_lm/block_lm_diagram.png`：生成的底图；最终校正以 PPT 为准。
- `data/presentations/block_lm/block_lm_explained.pptx`：主图页及可编辑说明页。
- `data/presentations/block_lm/block_lm_explained.pdf`：已渲染的两页 PDF。
- `data/presentations/block_lm/slide-1.png`：含最终校正的主图，2000 × 1125。
- `data/presentations/block_lm/slide-2.png`：算法说明页，2000 × 1125。
- `build_deck.py`：PPT 打包脚本，依赖 python-pptx。左侧主图为栅格图；右侧局部图、chi2 公式和第二页说明可编辑。

## 检查记录

底图经过独立视觉审查，发现上下局部图测量点约 8 像素漂移、chi2
公式分量下标不明确。图像工具修订因服务限额失败，改在 PPT 中用原生
图形重画两张局部图（共享同一原点相对坐标与轴长），用原生文本替换公式。
已重新生成并读取 PPT，检查两页、原生修正文字和全部元素的页面边界。
首次 Windows LibreOffice 在沙箱内报 `UtilBindVsockAnyPort: socket failed`。
续作时经执行审批，在沙箱外使用实际发行版 `AlmaLinux-9` 的 UNC 路径成功渲染。
首轮渲染发现公式遮挡候选标题、第二页卡片文字触底；已重新布局右侧说明区、
增加卡片高度并重渲染两页。最终渲染无文字遮挡或溢出，两幅局部图使用完全相同的
测量点相对坐标。PPT 结构、备注、图形边界和 PNG 尺寸检查通过；文件哈希见
`data/presentations/block_lm/verification.json`。
原始 PNG 底图保留生成时内容，展示或插入其他 PPT 应使用最终的 `slide-1.png`。

三个面是同一个 sensor 的不同位姿。图中 k+1 表示候选，只有接受后才成为
下一迭代；右侧白点表示全局预测交点的局部坐标。单条 hit 残差减小仅为示意，
不保证每条 hit 都改善，也不保证求解后的 target 面与 nominal 重合。
Nominal target 对照不进入该块残差；非 target nominal 几何仍提供固定径迹的参考。

## 源码依据

- `scripts/ml_alignment_autograd.py::load_joint_geometry`
- `scripts/ml_alignment_autograd.py::apply_alignment_torch`
- `scripts/ml_alignment_autograd.py::_sensor_residual_data`
- `scripts/ml_alignment_autograd.py::_block_lm_linearized_block`
- `scripts/ml_alignment_autograd.py::regularize_fisher_matrix`
- `scripts/ml_alignment_autograd.py::run_block_lm_alignment`
- `docs/handoff/2026-09-08-block-lm-convergence-contract.md`

制作时 HEAD 为 `a4dd1e1447555ca08b445bb52f80741c67d6236b`，工作区已有未提交修改。
本任务仅制作算法说明，未运行算法实验、未修改求解器。
