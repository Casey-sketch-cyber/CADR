# T8b 控制器计时

在新的 v2 评测配置中加入 `"measure_controller_timing": true`。IQL/CADR
和 Diffusion Policy/CADR-DP 的冻结权重评测都读这一配置。在独立输出目录执行，
评测结果用 `inference_ms_p50` 和 `inference_ms_p95` 报告每个方法、场景和种子的
控制步耗时中位数及第 95 百分位数，单位为毫秒。

每步以单调时钟测量 `controller.act`。CUDA 模式在读开始时间之前同步指定设备，
在读结束时间之前再次同步同一设备，计入本次控制器提交的 GPU 工作和等待时间。
用 `"device": "cpu"` 时采用同样的壁钟边界，省去 CUDA 同步。计时包含控制器内部的
时延推断、状态重建及策略推理，环境推进、信道模拟、重置和日志写入在计时范围之外。
Diffusion Policy 的动作块生成与缓存动作读取都进入控制步分布，首步也进入统计。

`controller_timing` 记录测量设备、是否进行 CUDA 同步和采样步数。
`controller_timing_context` 记录最大估计时延和 CADR 动力学集成大小。论文开销对照
采用 `max_delay=32` 和 5 成员的冻结 CADR 动力学权重。同一对照使用相同硬件、
场景、种子和步数预算。已有 v1 配置继续以原方式计时和保存结果。历史 v1 的耗时
字段来自原计时边界，T8b 的同步开销需用新配置实测。
