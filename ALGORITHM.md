# Cloud Cost Optimization Algorithm

## Main Algorithm: Deep Q-Network (DQN) for Cloud Service Selection

**Require:** Environment `env`, workload type `w`, training timesteps `T`, learning rate `α`, discount factor `γ`, exploration schedule `ε(t)`, replay buffer size `B`, batch size `b`, target network update interval `τ`, Q-network `Q(s,a;θ)`, target network `Q(s,a;θ⁻)`;

1: **Initialize** environment with workload type `w`
2: **Initialize** Q-network `Q(s,a;θ)` with random weights `θ`
3: **Initialize** target network `Q(s,a;θ⁻)` with weights `θ⁻ = θ`
4: **Initialize** replay buffer `D` with capacity `B`
5: **Initialize** step counter `t = 0`
6: **for** episode `e = 1` to `E` **do**
7:   **Initialize** state `s₀` by resetting environment
8:   **for** step `t = 0` to `T_episode` **do**
9:     **Select** action `a_t` using ε-greedy policy:
10:      **if** `rand() < ε(t)` **then**
11:        `a_t ← random_action()`
12:      **else**
13:        `a_t ← argmax_a Q(s_t, a; θ)`
14:      **end if**
15:    **Execute** action `a_t` in environment:
16:      **Decode** action: `(service_type, scale_action) ← decode(a_t)`
17:      **Handle** scaling for selected service:
18:        **if** `scale_action == 0` **then**
19:          `instances[service_type] ← max(0, instances[service_type] - 1)`
20:        **else if** `scale_action == 2` **then**
21:          `pending[service_type].append(startup_time[service_type])`
22:        **end if**
23:      **Process** pending instances:
24:        **for** each service `s` **do**
25:          **for** each pending instance `p` in `pending[s]` **do**
26:            **if** `p <= 0` **then**
27:              `instances[s] ← instances[s] + 1`
28:              **Remove** `p` from `pending[s]`
29:            **else**
30:              `p ← p - 1`
31:            **end if**
32:          **end for**
33:        **end for**
34:      **Handle** service interruptions:
35:        **for** each service `s` **do**
36:          **for** each instance `i` in `instances[s]` **do**
37:            **if** `rand() > reliability[s]` **then**
38:              `instances[s] ← instances[s] - 1`
39:              `interruptions[s] ← interruptions[s] + 1`
40:            **end if**
41:          **end for**
42:        **end for**
43:      **Calculate** current metrics:
44:        `demand_t ← workload[t]`
45:        `capacity_t ← Σ(instances[s] × capacity[s])`
46:        `utilization_t ← demand_t / capacity_t` **if** `capacity_t > 0` **else** `1.0`
47:        `latency_t ← calculate_latency(utilization_t)`
48:      **Calculate** costs:
49:        `total_cost ← 0`
50:        **for** each service `s` **do**
51:          **if** `instances[s] > 0` **then**
52:            `service_demand ← demand_t × (instances[s] × capacity[s] / capacity_t)`
53:            `cost[s] ← pricing_model[s](instances[s], service_demand, duration=1, time=t)`
54:            `total_cost ← total_cost + cost[s]`
55:          **end if**
56:        **end for**
57:      **Calculate** availability:
58:        **for** each service `s` **do**
59:          **if** `instances[s] > 0` **then**
60:            `availability[s] ← 1 - (interruptions[s] / instances[s])`
61:          **else**
62:            `availability[s] ← 1.0`
63:          **end if**
64:        **end for**
65:      **Check** SLA violations:
66:        `sla_violation ← 1` **if** `latency_t > latency_target` **else** `0`
67:        `availability_violation ← 1` **if** `min(availability) < availability_target` **else** `0`
68:      **Calculate** reward:
69:        `cost_term ← -total_cost`
70:        **if** `sla_violation == 1` **then**
71:          `latency_excess ← latency_t - latency_target`
72:          `latency_penalty ← -sla_penalty - (latency_excess / 100.0)`
73:        **else**
74:          `latency_penalty ← 0`
75:        **end if**
76:        `availability_penalty ← -availability_penalty × availability_violation`
77:        **if** `utilization_t >= high_load_threshold` **and** `sla_violation == 0` **and** `availability_violation == 0` **then**
78:          `high_load_bonus ← bonus_value`
79:        **else**
80:          `high_load_bonus ← 0`
81:        **end if**
82:        **if** `capacity_t == 0` **and** `demand_t > 0` **then**
83:          `zero_capacity_penalty ← -50.0 - (demand_t × 0.1)`
84:        **else**
85:          `zero_capacity_penalty ← 0`
86:        **end if**
87:        `r_t ← cost_term + latency_penalty + availability_penalty + high_load_bonus + zero_capacity_penalty`
88:      **Observe** next state `s_{t+1}`:
89:        `s_{t+1} ← [demand_t, utilization_t, latency_t, instances[], prices[]]`
90:      **Store** transition `(s_t, a_t, r_t, s_{t+1}, done)` in replay buffer `D`
91:    **Update** Q-network:
92:      **if** `|D| >= b` **then**
93:        **Sample** batch `{(s_i, a_i, r_i, s'_i, done_i)}` of size `b` from `D`
94:        **Compute** target Q-values:
95:          **for** each transition `(s_i, a_i, r_i, s'_i, done_i)` **do**
96:            **if** `done_i == True` **then**
97:              `y_i ← r_i`
98:            **else**
99:              `y_i ← r_i + γ × max_{a'} Q(s'_i, a'; θ⁻)`
100:            **end if**
101:          **end for**
102:        **Update** Q-network parameters:
103:          `θ ← θ - α × ∇_θ Σ_i (Q(s_i, a_i; θ) - y_i)²`
104:      **end if**
105:    **Update** target network:
106:      **if** `t mod τ == 0` **then**
107:        `θ⁻ ← θ`
108:      **end if**
109:    **Update** exploration rate: `ε(t) ← decay(ε_initial, ε_final, t)`
110:    **Update** step counter: `t ← t + 1`
111:    **if** `done == True` **then**
112:      **break**
113:    **end if**
114:  **end for**
115: **end for**

## Environment Step Algorithm

**Require:** Current state `s_t`, action `a_t`, environment parameters;

1: **Decode** action `a_t` to `(service_type, scale_action)`
2: **Apply** scaling action to selected service
3: **Process** pending instances (decrement startup timers, activate ready instances)
4: **Handle** service interruptions based on reliability
5: **Calculate** current demand `demand_t` from workload
6: **Calculate** total capacity `capacity_t` from active instances
7: **Calculate** utilization `utilization_t = demand_t / capacity_t`
8: **Calculate** latency `latency_t` based on utilization:
9:   **if** `utilization_t <= 0.6` **then**
10:     `latency_t ← 120.0`
11:   **else if** `utilization_t <= 0.8` **then**
12:     `latency_t ← 120.0 + (utilization_t - 0.6) × 300.0`
13:   **else**
14:     `latency_t ← 180.0 + (utilization_t - 0.8) × 1000.0`
15:   **end if**
16: **Calculate** service costs using pricing models:
17:   **for** each service `s` **do**
18:     **if** `instances[s] > 0` **then**
19:       `service_demand ← demand_t × (capacity[s] / capacity_t)`
20:       `cost[s] ← pricing_function[s](instances[s], service_demand, duration, time)`
21:     **end if**
22:   **end for**
23: **Calculate** total cost `total_cost = Σ cost[s]`
24: **Calculate** service availability based on interruptions
25: **Check** SLA violations (latency and availability)
26: **Calculate** reward using reward function
27: **Update** environment history
28: **Return** `(s_{t+1}, r_t, done, info)`

## Reward Calculation Algorithm

**Require:** Total cost `C`, latency `L`, utilization `U`, availability `A`, availability violation `V_a`, capacity `Cap`, demand `D`;

1: **Initialize** reward components:
2:   `cost_term ← -C`
3:   `latency_penalty ← 0`
4:   `availability_penalty ← 0`
5:   `high_load_bonus ← 0`
6:   `zero_capacity_penalty ← 0`
7: **Check** latency violation:
8:   **if** `L > latency_target` **then**
9:     `latency_excess ← L - latency_target`
10:     `latency_penalty ← -sla_penalty - (latency_excess / 100.0)`
11:   **end if**
12: **Check** availability violation:
13:   `availability_penalty ← -availability_penalty_value × V_a`
14: **Check** high load bonus:
15:   **if** `U >= high_load_threshold` **and** `L <= latency_target` **and** `V_a == 0` **then**
16:     `high_load_bonus ← bonus_value`
17:   **end if**
18: **Check** zero capacity penalty:
19:   **if** `Cap == 0` **and** `D > 0` **then**
20:     `zero_capacity_penalty ← -50.0 - (D × 0.1)`
21:   **end if**
22: **Calculate** total reward:
23:   `R ← cost_term + latency_penalty + availability_penalty + high_load_bonus + zero_capacity_penalty`
24: **Return** `(R, {cost_term, latency_penalty, availability_penalty, high_load_bonus, zero_capacity_penalty})`

## State Observation Algorithm

**Require:** Current time step `t`, workload `W`, service instances `I`, services `S`;

1: **Get** current demand: `demand ← W[t]`
2: **Calculate** total capacity:
3:   `capacity ← 0`
4:   **for** each service `s` in `S` **do**
5:     `capacity ← capacity + I[s] × capacity[s]`
6:   **end for**
7: **Calculate** utilization:
8:   **if** `capacity > 0` **then**
9:     `utilization ← demand / capacity`
10:   **else**
11:     `utilization ← 1.0`
12:   **end if**
13: **Get** current latency: `latency ← current_latency`
14: **Get** service instances: `instances ← [I[s] for s in S]`
15: **Get** service prices:
16:   `prices ← []`
17:   **for** each service `s` in `S` **do**
18:     `price ← pricing_model[s](1, 0, 1.0, t)`
19:     `prices.append(price)`
20:   **end for**
21: **Construct** state vector:
22:   `state ← [demand, utilization, latency, instances[], prices[]]`
23: **Return** `state`

## Evaluation Algorithm

**Require:** Trained model `M`, environment `env`, number of episodes `E`;

1: **Initialize** results dictionary `results`
2: **for** episode `e = 1` to `E` **do**
3:   **Reset** environment: `s_0 ← env.reset()`
4:   **Initialize** episode metrics:
5:     `episode_reward ← 0`
6:     `episode_cost ← 0`
7:     `episode_violations ← 0`
8:   **for** step `t = 0` to `T_episode` **do**
9:     **Select** action: `a_t ← M.predict(s_t, deterministic=True)`
10:     **Execute** action: `(s_{t+1}, r_t, done, info) ← env.step(a_t)`
11:     **Update** metrics:
12:       `episode_reward ← episode_reward + r_t`
13:       `episode_cost ← episode_cost + info['total_cost']`
14:       `episode_violations ← episode_violations + info['sla_violation']`
15:     **if** `done == True` **then**
16:       **break**
17:     **end if**
18:   **end for**
19:   **Store** episode results in `results`
20: **end for**
21: **Calculate** summary statistics:
22:   `avg_reward ← mean(results['rewards'])`
23:   `avg_cost ← mean(results['costs'])`
24:   `avg_violations ← mean(results['violations'])`
25: **Return** `results` and summary statistics

## Comparison Algorithm

**Require:** Strategies `S = {s_1, s_2, ..., s_n}`, workload types `W = {w_1, w_2, ..., w_m}`, episodes `E`;

1: **Initialize** comparison results `comparison_results`
2: **for** each workload type `w` in `W` **do**
3:   **Create** environment `env` with workload type `w`
4:   **for** each strategy `s` in `S` **do**
5:     **Run** evaluation algorithm with `(s, env, E)`
6:     **Store** results in `comparison_results[w][s]`
7:   **end for**
8: **end for**
9: **Calculate** cross-workload statistics:
10:   **for** each strategy `s` in `S` **do**
11:     `avg_cost[s] ← mean(comparison_results[*][s]['cost'])`
12:     `avg_violations[s] ← mean(comparison_results[*][s]['violations'])`
13:   **end for**
14: **Generate** comparison report and visualizations
15: **Return** `comparison_results`
