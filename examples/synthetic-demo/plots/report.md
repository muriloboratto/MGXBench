# MGXBench Performance Report

- Only numerically validated measurements are plotted.
- Bars show means ± population standard deviation. CI95 is a seeded percentile bootstrap of independent launches (2,000 resamples); small samples give uncertain intervals.
- GFLOP/s is effective application throughput: 2 N³ / elapsed time. No hardware-counter or isolated-kernel claim.
- Identical environment, label and internal-loop count are required for comparisons. Repeated matching runs pool raw launch samples.
- 1D and 2D timed regions retain their original algorithm-specific scope. Equal internal-loop count does not make their communication work identical.
- Matrix-size curves describe workload growth; strong scaling fixes N. Weak scaling for dense MM holds N³/p approximately constant, not N²/p.
- Kernel and application totals do not identify concurrent phases. This release provides benchmark summaries and kernel statistics only.

![325c179628_N1024_p1_execution_time.png](325c179628_N1024_p1_execution_time.png)

![325c179628_N1024_p1_throughput.png](325c179628_N1024_p1_throughput.png)

![325c179628_N1024_p1_speedup.png](325c179628_N1024_p1_speedup.png)

![325c179628_N1024_p1_1d_time_distribution.png](325c179628_N1024_p1_1d_time_distribution.png)

![325c179628_N1024_p1_2d_time_distribution.png](325c179628_N1024_p1_2d_time_distribution.png)

![325c179628_N1024_p4_execution_time.png](325c179628_N1024_p4_execution_time.png)

![325c179628_N1024_p4_throughput.png](325c179628_N1024_p4_throughput.png)

![325c179628_N1024_p4_speedup.png](325c179628_N1024_p4_speedup.png)

![325c179628_N1024_p4_1d_time_distribution.png](325c179628_N1024_p4_1d_time_distribution.png)

![325c179628_N1024_p4_2d_time_distribution.png](325c179628_N1024_p4_2d_time_distribution.png)

![325c179628_N2048_p1_execution_time.png](325c179628_N2048_p1_execution_time.png)

![325c179628_N2048_p1_throughput.png](325c179628_N2048_p1_throughput.png)

![325c179628_N2048_p1_speedup.png](325c179628_N2048_p1_speedup.png)

![325c179628_N2048_p1_1d_time_distribution.png](325c179628_N2048_p1_1d_time_distribution.png)

![325c179628_N2048_p1_2d_time_distribution.png](325c179628_N2048_p1_2d_time_distribution.png)

![325c179628_N2048_p4_execution_time.png](325c179628_N2048_p4_execution_time.png)

![325c179628_N2048_p4_throughput.png](325c179628_N2048_p4_throughput.png)

![325c179628_N2048_p4_speedup.png](325c179628_N2048_p4_speedup.png)

![325c179628_N2048_p4_1d_time_distribution.png](325c179628_N2048_p4_1d_time_distribution.png)

![325c179628_N2048_p4_2d_time_distribution.png](325c179628_N2048_p4_2d_time_distribution.png)

![325c179628_1d_p1_size_scalability.png](325c179628_1d_p1_size_scalability.png)

![325c179628_1d_p4_size_scalability.png](325c179628_1d_p4_size_scalability.png)

![325c179628_2d_p1_size_scalability.png](325c179628_2d_p1_size_scalability.png)

![325c179628_2d_p4_size_scalability.png](325c179628_2d_p4_size_scalability.png)

![325c179628_1d_N1024_strong_scaling.png](325c179628_1d_N1024_strong_scaling.png)

![325c179628_1d_N2048_strong_scaling.png](325c179628_1d_N2048_strong_scaling.png)

![325c179628_2d_N1024_strong_scaling.png](325c179628_2d_N1024_strong_scaling.png)

![325c179628_2d_N2048_strong_scaling.png](325c179628_2d_N2048_strong_scaling.png)

![46cc8846ce_N1024_p1_execution_time.png](46cc8846ce_N1024_p1_execution_time.png)

![46cc8846ce_N1024_p1_throughput.png](46cc8846ce_N1024_p1_throughput.png)

![46cc8846ce_N1024_p1_speedup.png](46cc8846ce_N1024_p1_speedup.png)

![46cc8846ce_N1024_p1_1d_time_distribution.png](46cc8846ce_N1024_p1_1d_time_distribution.png)

![46cc8846ce_N2048_p8_execution_time.png](46cc8846ce_N2048_p8_execution_time.png)

![46cc8846ce_N2048_p8_throughput.png](46cc8846ce_N2048_p8_throughput.png)

![46cc8846ce_N2048_p8_speedup.png](46cc8846ce_N2048_p8_speedup.png)

![46cc8846ce_N2048_p8_1d_time_distribution.png](46cc8846ce_N2048_p8_1d_time_distribution.png)

![46cc8846ce_1d_MMM_weak_scaling.png](46cc8846ce_1d_MMM_weak_scaling.png)

![46cc8846ce_1d_NNN_weak_scaling.png](46cc8846ce_1d_NNN_weak_scaling.png)