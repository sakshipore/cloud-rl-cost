## Comparison of Heuristic, Cost-Only DQN, and Proposed Framework

The table below summarizes the steady-workload results for the three approaches, using the available metrics from `research_outputs_final/steady/comparison_table.csv`. Metrics not reported in the data are marked as `N/A`.

| Approach                    | Operational Cost (Total) | SLA Violation Rate | Latency (Avg, ms) | Energy Consumption | Throughput |
|----------------------------|--------------------------|--------------------|-------------------|--------------------|------------|
| Heuristic (Traditional)    | $641.91                  | 1.78%              | 128.69            | N/A                | N/A        |
| Cost-Only DQN (VpQ-based)  | $230.99                  | 97.90%             | 729.06            | N/A                | N/A        |
| Proposed Framework (DQN)   | **$29.15**               | **1.00%**          | 123.13            | N/A                | N/A        |

Values are taken from:

- `traditional` → Heuristic (Traditional)
- `vpq_inspired` → Cost-Only DQN (VpQ-based baseline RL)
- `dqn` → Proposed Framework (SLA-aware DQN)

