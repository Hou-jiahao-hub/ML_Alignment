# 2026-10-15 pT=100 GeV ACTS-Alignment vs Schur-LM

- 样本：g3 20k 事件，pT 100 GeV，材料开。误校准 seed 20261015，两种方法共用（SHA 3c5ad160…）。
- 耗时（单线程，资源对等）：ACTS 16176 s，9 次迭代，每次约 30 min，时间主要花在稠密 13608² 求逆和 fullPivLu 上。Schur 544 s（solve 481 s），加上 misaligned 重建为 588 s。Schur 3 线程 319 s，导出几何与单线程逐位相同。
- 精度：几何 RMS 每格都是 Schur 更低，但这完全来自 1–2 个 hit 的 sensor。剔除 hit ≤6 的 sensor 后两者逐格一致，中位数相同，重建 χ²/ndf 和 meas−truth 差异 ≤0.1%。
- 文档：优化日志 Iteration 48，交接 docs/handoff/2026-10-15-pt100-acts-vs-schur.md。
