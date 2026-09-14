# SMT seed 与 measurement refit 汇报素材

**六页正文＋十四页附录；只交付文案与图稿，不生成 PPTX。**

## 从这里使用

1. [正文逐页文案 P1–P6](slides.md)：页面标题、可直接上屏的文字和公式、布局、讲解备注。
2. [公式附录 A1–A14](appendix.md)：完整数学推导与实现边界。
3. [原图核验表](verification.md)：原图问题、修正理由与代码对应。
4. [检查记录](qa.md)、[源码版本及哈希](source_snapshot.json)：用于复核和归档。

## 图稿与页码

SVG 图稿随本目录一起入库，位于 [assets/](assets/)，是插入 PPT 时建议使用的矢量源；Mermaid 图源和版面脚本保留在 `diagrams/`。SVG 插入后的可编辑程度取决于 PowerPoint 版本，不能保证等同于原生流程图形状。

PNG（宽 3840 像素）和九图总览 `contact_sheet.png` 是派生产物，按本仓库 `data/` 的惯例不入库。需要位图时用下面「复现图稿」一节的命令重新生成到 `data/presentations/seed_refit/`。

| 页码 | 推荐图稿 | 图源／补充 |
|---|---|---|
| P1 | [SVG](assets/p1_batches.svg) | [组批图源](diagrams/p1_batches.mmd) |
| P2 | [Gauge SVG](assets/p2_gauge.svg) | [参数映射 SVG](assets/p2_parameters.svg) |
| P3 | [紧凑 seed SVG](assets/p3_seed.svg) | [详细版 SVG](assets/p3_seed_detailed.svg) · [图源](diagrams/p3_seed.mmd) |
| P4 | [紧凑 refit SVG](assets/p4_refit.svg) | [详细版 SVG](assets/p4_refit_detailed.svg) · [图源](diagrams/p4_refit.mmd) |
| P5 | [验收 SVG](assets/p5_gate.svg) | [图源](diagrams/p5_gate.mmd) |
| P6 | [输出 SVG](assets/p6_output.svg) | [图源](diagrams/p6_output.mmd) |
| A6 | [最近点／求交对比 SVG](assets/a6_closest_vs_intersection.svg) | 也可截取左侧用于 P3 |
| A10／P4 | [局部预测 SVG](assets/a10_intersection.svg) | 求交、坐标变换和测量分量 |

正文图用来说明算法，不是拟合数据或性能结果。详细版供放大审阅；紧凑版保留正文所需的更新与失败回路。所有图都需结合该页短文案使用，尤其是非驻点比例、有效性门限和浮点精度说明。

## 编排建议

- 保留 P1–P6 编号顺序，在整体 PPT 内按所在章节重新编号即可。
- 输入、目标、输出仅在 P3、P4 页顶部简述；不另设背景介绍或算法核验正文页。
- P2 必须保留“同一曲线的原点冗余→规范固定→五维优化”的解释，不把曲率缩放当作降维原因。
- 公式使用 PPT 公式编辑器；Markdown 中的 LaTeX 是可编辑源。每页的讲解备注不必上屏。
- 附录表格、长推导允许比正文更高信息密度。A14 阈值表在报告时按需要定位，不逐行宣读。
- 原图的几何更新旋转矩阵修正见核验记录；不占本章节六页正文。

## 复现图稿

在项目根目录，使用 conda `pytorch` Python 执行：

```bash
python docs/presentations/seed_refit/diagrams/render_schematics.py
```

该脚本重写 `assets/` 下五份版面 SVG（`p2_gauge`、`p3_seed`、`p4_refit`、`a6_closest_vs_intersection`、`a10_intersection`）。

Mermaid 使用本机已有 `mmdc` 和 `diagrams/mermaid_config.json` 渲染其余六份，同样输出到 `assets/`；P3、P4 的 Mermaid 输出命名为 `*_detailed.svg`，不要覆盖由版面脚本生成的紧凑版。例：

```bash
mmdc -i docs/presentations/seed_refit/diagrams/p3_seed.mmd \
  -o docs/presentations/seed_refit/assets/p3_seed_detailed.svg \
  -c docs/presentations/seed_refit/diagrams/mermaid_config.json
```

本机 Chromium 需 `--no-sandbox` 配置以便本地渲染；这是构建环境要求，不是算法步骤。PNG 导出脚本从 `assets/` 读取 SVG，把位图写到 `data/presentations/seed_refit/`，并刷新 `assets/render_checks.json`。它接受 Puppeteer 模块和 Chromium 的本地路径：

```bash
node docs/presentations/seed_refit/diagrams/export_png.cjs \
  /path/to/puppeteer /path/to/chrome
```

本次使用现有 Mermaid CLI 11.16.0 与本地 Chromium，无网络字体、无新依赖安装。没有覆盖原有 PPT 或修改算法文件。
