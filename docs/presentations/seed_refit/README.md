# SMT seed 与 measurement refit 汇报素材

**正文编号七页（P3、P4 各以三页渐进图展开，P7 用单行流程带，上屏十一页）＋十四页附录；只交付文案与图稿，不生成 PPTX。**

## 从这里使用

1. [正文逐页文案 P1–P7](slides.md)：页面标题、可直接上屏的文字和公式、布局、讲解备注。
2. [公式附录 A1–A14](appendix.md)：完整数学推导与实现边界。
3. [原图核验表](verification.md)：原图问题、修正理由与代码对应。
4. [检查记录](qa.md)、[源码版本及哈希](source_snapshot.json)：用于复核和归档。

## 图稿与页码

SVG 图稿随本目录一起入库，位于 [assets/](assets/)，是插入 PPT 时建议使用的矢量源；Mermaid 图源和版面脚本保留在 `diagrams/`。SVG 插入后的可编辑程度取决于 PowerPoint 版本，不能保证等同于原生流程图形状。

PNG（宽 3840 像素）和图稿总览 `contact_sheet.png` 是派生产物，按本仓库 `data/` 的惯例不入库。需要位图时用下面「复现图稿」一节的命令重新生成到 `data/presentations/seed_refit/`。

| 页码 | 推荐图稿 | 图源／补充 |
|---|---|---|
| P1 | [SVG](assets/p1_batches.svg) | [组批图源](diagrams/p1_batches.mmd) |
| P2 | [Gauge SVG](assets/p2_gauge.svg) | [参数映射 SVG](assets/p2_parameters.svg) |
| P3a／b／c | 渐进三页：[主干](assets/p3_seed_s1.svg) · [内循环](assets/p3_seed_s2.svg) · [收敛与验收](assets/p3_seed_s3.svg) | 附录备查：[单页版](assets/p3_seed.svg) · [详细版](assets/p3_seed_detailed.svg) · [图源](diagrams/p3_seed.mmd) |
| P4a／b／c | 渐进三页：[主干](assets/p4_refit_s1.svg) · [内循环](assets/p4_refit_s2.svg) · [收敛与验收](assets/p4_refit_s3.svg) | 附录备查：[单页版](assets/p4_refit.svg) · [详细版](assets/p4_refit_detailed.svg) · [图源](diagrams/p4_refit.mmd) |
| P5 | [验收 SVG](assets/p5_gate.svg) | [图源](diagrams/p5_gate.mmd) |
| P6 | [输出 SVG](assets/p6_output.svg) | [图源](diagrams/p6_output.mmd) |
| P7 | [流程带 SVG](assets/p7_blocklm_band.svg) | 单行流程带，与 P3／P4 上屏所用的带状图同风格；由 `render_schematics.blocklm_band` 生成 |
| A6 | [最近点／求交对比 SVG](assets/a6_closest_vs_intersection.svg) | 也可截取左侧用于 P3 |
| A10／P4 | [局部预测 SVG](assets/a10_intersection.svg) | 求交、坐标变换和测量分量 |

正文图用来说明算法，不是拟合数据或性能结果。详细版供放大审阅；单页版保留完整的更新与失败回路。所有图都需结合该页短文案使用，尤其是非驻点比例、有效性门限和浮点精度说明。

P3、P4 的三页图是同一版面的三个阶段：主干框、菱形与出口框在三页中坐标完全相同，只切换高亮与淡出，因此必须按 a→b→c 连续放映才成立，抽单张使用会丢掉「折叠块在下一页展开」的线索。它们按 1600×900 绘制，框内正文 34 px，需占页宽不少于 90% 才能达到 18 pt 的上屏字号。

## 编排建议

- 保留 P1–P7 编号顺序，在整体 PPT 内按所在章节重新编号即可。P7 讲固定径迹后的目标层几何优化，必须排在 P6 之后。P3、P4 展开为 a→b→c 三页，顺序不可打乱，中间不要插入其他内容。
- P3、P4 的六页图各自整幅上屏，不与文字并排；该页文案放演讲者备注。每页图右上角的 1/3、2/3、3/3 用于提示听众当前在哪一层。
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

该脚本重写 `assets/` 下十二份版面 SVG：五份原有图（`p2_gauge`、`p3_seed`、`p4_refit`、`a6_closest_vs_intersection`、`a10_intersection`）、六份渐进图（`p3_seed_s1`–`s3`、`p4_refit_s1`–`s3`）和一份流程带（`p7_blocklm_band`）。流程带用 `Canvas.curve` 画弯箭头、只落文字不画框，坐标写在 `blocklm_band()` 内部，与渐进图的共享槽位互不影响。渐进图的槽位坐标在脚本顶部的 `T1`–`TX`、`I1`–`I6`、`FOLD` 常量里集中定义，三个阶段共用同一套，不要在各阶段函数里另写坐标——那会破坏连播对齐。

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
