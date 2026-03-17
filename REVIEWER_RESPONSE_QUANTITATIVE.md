# Reviewer Response: Quantitative Results and Clarifications

This document provides precise answers to reviewer comments regarding quantitative results, statistical significance, reward function details, baseline implementation, references, and figure data.

---

## 1. Exact Quantitative Results (Comment [a])

Reviewer requested numbers such as “reduces SLA violations by X%”.

### From experiments (steady workload, 5 evaluation episodes)

| Metric | Traditional (heuristics) | VpQ-inspired (baseline RL) | Proposed (DQN SLA-aware) |
|--------|---------------------------|----------------------------|---------------------------|
| **SLA violation rate** | 1.78% (0.01775) | **97.9%** (0.979) | **1.0%** (0.01) |
| **Total cost** | $641.91 | $230.99 | **$29.15** |
| **Avg latency (ms)** | 128.69 | 729.06 | 123.13 |

### Summary for rebuttal

- **Average SLA violation rate**
  - **Baseline RL (VpQ-inspired):** **97.9%** (mean over 5 episodes; std_sla_violation_rate ≈ 0.016).
  - **Proposed RL (DQN):** **1.0%** (mean; std_sla_violation_rate ≈ 0.0032).

- **SLA violation reduction (proposed vs baseline RL):**  
  **97.9% → 1.0%** → approximately **99% relative reduction** (e.g. “reduces SLA violation rate from 97.9% to 1.0%”).

- **Average cost reduction**
  - **vs baseline RL:** Proposed $29.15 vs VpQ $230.99 → **~7.9× lower cost** (~87% reduction).
  - **vs heuristics:** Proposed $29.15 vs traditional $641.91 → **~22× lower cost** (~95% reduction).

Example phrasing: “reduces SLA violations from ~98% to ~1%” and “~8× lower cost than baseline RL, ~22× lower than heuristics.”

---

## 2. Statistical Significance (Comment [g])

### What was done

- **Multiple evaluation runs:** Yes. Evaluation uses **n_episodes** (e.g. 5) with **different reset seeds** per episode: `env.reset(seed=42 + episode)` in `rl/evaluate.py`, `baselines/comprehensive_comparison.py`, and `evaluation/comprehensive_eval.py`.
- **Mean ± standard deviation:** Yes. Summary statistics use `np.mean` and `np.std` over episodes for cost, SLA violation rate, latency, etc. (e.g. `_calculate_summary_stats` in `rl/evaluate.py`, `_aggregate_metrics` in `baselines/comprehensive_comparison.py`). These are stored in `detailed_results.json` and used in plots (e.g. `yerr=std_costs` in `visualization/academic_plots.py`).
- **Confidence intervals:** **No.** Not computed in the codebase.
- **t-tests or similar:** **No.** No significance tests between methods are implemented.

### Runs per method

- **Per workload:** **5 episodes** (configurable `n_episodes=5` in the research evaluation).
- **Training:** A **single training seed** (e.g. 42) is used; there are no multiple training seeds. Variance reported is across evaluation episodes (different episode seeds), not across training runs.

### For the “statistical significance” subsection

You can state that you report **mean ± standard deviation over 5 evaluation episodes** and that significance testing (e.g. t-tests) and confidence intervals were not implemented; adding them would strengthen the revision.

---

## 3. Reward Function Details (for “reward sensitivity and stability”)

### Actual reward equation (in code)

From `envs/enhanced_cloud_env.py`, `_calculate_reward`:

```
R = cost_term + latency_penalty + availability_penalty + high_load_bonus + zero_capacity_penalty
```

Where:

- **cost_term** = `-total_cost` (cost coefficient **1**; we minimize cost).
- **latency_penalty:**  
  - If `latency > latency_target`:  
    `latency_penalty = -sla_penalty - (latency_excess / 100.0)`  
    with `latency_excess = latency - latency_target`.  
  - Else: `0`.
- **availability_penalty** = `-availability_penalty * availability_violation`  
  (`availability_violation` is 0 or 1).
- **high_load_bonus** = `high_load_bonus` if `utilization >= high_load_threshold` and no SLA/availability violation; else `0`.
- **zero_capacity_penalty** = `-50.0 - (current_demand * 0.1)` if `total_capacity == 0` and `current_demand > 0`; else `0`.

### Parameter values (from `envs/enhanced_cloud_env.py`)

| Parameter | Symbol / name in code | Value | Role |
|-----------|------------------------|------|------|
| Cost | (implicit weight 1) | 1 | `cost_term = -total_cost` |
| Latency (SLA) | `sla_penalty` | **5.0** | Base penalty when latency > target; plus progressive term `latency_excess/100` |
| Availability | `availability_penalty` | **1.5** | Penalty per availability violation |
| Utilization bonus | `high_load_bonus` | **0.5** | Bonus when utilization ≥ threshold and SLA met |
| High-load threshold | `high_load_threshold` | **0.8** | Utilization ≥ 80% |
| Latency target | `latency_target` | **200** ms | SLA latency bound |
| Zero-capacity penalty | — | **-50.0 - demand×0.1** | Discourage zero capacity |

### How weights were set

- **Fixed manually** in code (no automatic tuning).
- `documentation/REWARD_FUNCTION_DESIGN.md` describes design rationale and older defaults (e.g. sla_penalty 2.0); the **current implemented** values are as in the table above.
- Weights are **not** formally derived from SLA values; they were chosen to balance cost and SLA in experiments.

For “reward sensitivity and stability,” state the equation and table above and note that weights are fixed and could be tuned or derived from SLA targets in future work.

---

## 4. Baseline RL Implementation

### What the baseline is

- **Same environment:** Yes. All methods (traditional, VpQ-inspired, DQN) use `EnhancedCloudCostGym` with the same `n_steps`, `workload_type`, and evaluation protocol.
- **Algorithm:** **Tabular Q-learning** with **discrete state** (binned demand, utilization, prices), not DQN. So: same environment, **different** algorithm (Q-learning vs DQN).
- **Reward:** **Cost-only.** The VpQ-inspired agent uses a **cost-only** reward. In `baselines/vpq_inspired.py`, during training it uses `cost_only_reward = -step_cost` (with step cost inferred from the env step). The environment still returns its full reward (cost + SLA penalties); the baseline **ignores** the SLA part and optimizes only cost.

### Explicit clarification

- **Algorithm:** VpQ-inspired **tabular Q-learning** (discrete state bins, ε-greedy, Q-table updates), not the full VpQ-learning paper formulation; “inspired” by that line of work.
- **Reward definition:** **R_baseline = -cost** (no latency or availability terms). Same env step, but the baseline agent uses only the cost component.

So the comparison is: **SLA-aware DQN** (full reward) vs **cost-only tabular Q-learning** (same env, cost-only reward).

---

## 5. Top-tier Journal References (Comment [h])

Whether you can add 2–3 references (e.g. IEEE TCC, IEEE TPDS, FGCS, ACM TOS) is a **journal policy and space** question, not something the codebase decides. If the editor allows a small number of additional references in the literature survey, you can add them; otherwise you may need to replace existing ones or ask the editor.

---

## 6. Figures and Tables – Raw Data

**Yes, there is raw or aggregated data behind the figures:**

- **Fig. 2 (cost):** From `compare_strategies` / `compare_all_approaches`. Each strategy has per-episode `total_cost`; summary is `mean` and `std`. Plots use `results[approach]["total_cost"]` and `std_cost` (e.g. `visualization/academic_plots.py`, `plot_cost_comparison` with `yerr=std_costs`). So the figure is backed by **real** cost data (mean ± std over episodes).
- **Fig. 3 (SLA):** Same pipeline; SLA violation rate (and optionally availability/latency) come from the same evaluation results and summary stats (e.g. `sla_violation_rate` mean/std). Again **real** data.
- **Fig. 5 (trade-off):** `_plot_performance_tradeoffs` / `plot_performance_tradeoffs` use the same `comparison_results` (cost and SLA violation rate per strategy). So the trade-off plot is also from **real** data, not conceptual.

### Where the data lives

- **Per-episode and summary:** In `detailed_results.json` (per strategy: episodes list and metrics like `total_cost`, `sla_violation_rate` per episode), and in the in-memory `comparison_results` that feed the plotters.
- **Tabular:** In `research_outputs_final/steady/comparison_table.csv` (and similar paths for other runs) with columns such as Total Cost, SLA Violation Rate, etc.

So you can state that Fig. 2, Fig. 3, and Fig. 5 are **generated from experimental comparison results**, with raw data available in `detailed_results.json` and summary tables in the CSV reports.

---

## Short Summary for Rebuttal

- **Numbers:** Baseline RL SLA violation rate ~97.9%, proposed ~1.0%; cost ~7.9× lower than baseline RL and ~22× lower than heuristics (steady workload, 5 episodes).
- **Stats:** Mean ± std over 5 evaluation episodes; no CIs or t-tests; single training seed.
- **Reward:** Equation and weights as in Section 3; weights fixed manually.
- **Baseline RL:** Tabular Q-learning, cost-only reward, same environment.
- **References:** Add if the journal allows.
- **Figures:** All backed by real comparison data; raw data in `detailed_results.json` and CSV comparison tables.
