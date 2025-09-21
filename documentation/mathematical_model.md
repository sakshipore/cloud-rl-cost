# Mathematical Model of Cloud RL Cost Optimization System

## 1. System Overview and Notation

### Core Sets and Entities

**Service Set:**
```
S = {s₁, s₂, s₃, s₄} = {EC2_OnDemand, EC2_Spot, Lambda, Fargate}
```

**Workload Types:**
```
W = {w₁, w₂, w₃, w₄} = {diurnal, steady, batch, bursty}
```

**Time Steps:**
```
T = {0, 1, 2, ..., T_max} where T_max = 1440 (minutes in a day)
```

**Action Space:**
```
A = {a₁, a₂, ..., a₁₂} = {(sᵢ, scale_j) | sᵢ ∈ S, scale_j ∈ {0, 1, 2}}
```
where scale_j represents: 0=scale_down, 1=no_change, 2=scale_up

**State Space:**
```
X = ℝ¹⁰ = [demand, utilization, latency, instances_s₁, instances_s₂, instances_s₃, instances_s₄, price_s₁, price_s₂, price_s₃, price_s₄]
```

## 2. Input and Output Definitions

### Input Set (I)
```
I = {
    workload_type: w ∈ W,
    n_steps: T_max ∈ ℕ,
    seed: s ∈ ℕ,
    services_config: C_s,
    pricing_models: P_s,
    sla_parameters: SLA
}
```

### Output Set (O)
```
O = {
    total_cost: c ∈ ℝ₊,
    sla_violations: v ∈ ℕ,
    service_usage: U ∈ ℝ₊⁴,
    performance_metrics: M ∈ ℝᵏ
}
```

## 3. Core Mathematical Functions

### 3.1 Workload Generation Function

**Diurnal Workload:**
```
D_diurnal(t) = 200 + 150·sin(2πt/T_max · 2) + Σᵢ αᵢ·spike_i(t) + ε(t)
```
where:
- `αᵢ ~ Uniform(200, 600)` (spike heights)
- `spike_i(t)` is a triangular spike function
- `ε(t) ~ N(0, 20²)` (Gaussian noise)

**Steady Workload:**
```
D_steady(t) = 300 + 50·sin(2πt/60) + ε(t)
```
where `ε(t) ~ N(0, 15²)`

**Batch Workload:**
```
D_batch(t) = Σᵢ βᵢ·I[tᵢ, tᵢ + dᵢ](t) + ε(t)
```
where:
- `βᵢ ~ Uniform(400, 800)` (batch intensities)
- `I[a,b]` is indicator function
- `dᵢ ~ Uniform(30, 120)` (batch durations)

**Bursty Workload:**
```
D_bursty(t) = 100 + Σᵢ γᵢ·exp(-(t - τᵢ)/σᵢ) + ε(t)
```
where:
- `γᵢ ~ Uniform(300, 700)` (burst heights)
- `τᵢ` are random burst times
- `σᵢ ~ Uniform(2, 10)` (decay rates)

### 3.2 Pricing Functions

**On-Demand Pricing:**
```
P_ondemand(instances, duration, t) = instances · 0.9 · (0.95 + 0.1·U(0,1)) · duration
```

**Spot Pricing:**
```
P_spot(instances, duration, t) = instances · 0.9 · (1 - discount) · duration
```
where `discount ~ Beta(2, 5) · 0.6 + 0.3` (30-90% discount)

**Serverless Pricing:**
```
P_lambda(requests, duration, t) = requests · 0.0000002 + duration · 0.9 · 0.1
```

**Container Pricing:**
```
P_fargate(instances, duration, t) = instances · 0.9 · 1.15 · duration
```

### 3.3 State Transition Function

```
f: X × A → X
f(x_t, a_t) = x_{t+1}
```

**State Update Components:**
```
demand_{t+1} = D(t+1)
utilization_{t+1} = demand_{t+1} / total_capacity_t
latency_{t+1} = L(utilization_{t+1})
instances_{s,t+1} = instances_{s,t} + scaling_action(a_t)
prices_{s,t+1} = P_s(1, 0, 1, t+1)
```

**Latency Function:**
```
L(u) = {
    120,                    if u ≤ 0.6
    120 + (u - 0.6) · 300,  if 0.6 < u ≤ 0.8
    180 + (u - 0.8) · 1000, if u > 0.8
}
```

### 3.4 Reward Function

```
R(x_t, a_t, x_{t+1}) = -(total_cost_t + penalty_t)
```

where:
```
total_cost_t = Σ_{s∈S} P_s(instances_{s,t}, demand_{s,t}, 1, t)
penalty_t = SLA_penalty · I[latency_t > latency_target]
```

**SLA Violation Indicator:**
```
I[latency > target] = {
    1, if latency > 200ms
    0, otherwise
}
```

## 4. Service Characteristics and Constraints

### 4.1 Service Capacity Constraints

```
∀s ∈ S: instances_s ≤ max_instances_s
```

where:
- `max_instances_EC2_OnDemand = 20`
- `max_instances_EC2_Spot = 20`
- `max_instances_Lambda = ∞`
- `max_instances_Fargate = 15`

### 4.2 Reliability Constraints

```
P(interruption_s) = 1 - reliability_s
```

where:
- `reliability_EC2_OnDemand = 0.999`
- `reliability_EC2_Spot = 0.95`
- `reliability_Lambda = 0.999`
- `reliability_Fargate = 0.999`

### 4.3 Capacity Calculation

```
total_capacity_t = Σ_{s∈S} instances_{s,t} · capacity_s
```

where:
- `capacity_EC2_OnDemand = 150` req/s
- `capacity_EC2_Spot = 150` req/s
- `capacity_Lambda = 100` req/s
- `capacity_Fargate = 120` req/s

## 5. Reinforcement Learning Model

### 5.1 Q-Function

```
Q_θ(s, a) = E[R_t + γ max_{a'} Q_θ(s', a') | s_t = s, a_t = a]
```

where:
- `θ` are neural network parameters
- `γ = 0.99` (discount factor)
- `R_t` is the reward at time t

### 5.2 Loss Function

```
L(θ) = E[(Q_θ(s, a) - y)²]
```

where:
```
y = r + γ max_{a'} Q_{θ'}(s', a')
```

and `θ'` are target network parameters.

### 5.3 Experience Replay

```
D = {(s_i, a_i, r_i, s'_i, done_i)}_{i=1}^N
```

**Sampling:**
```
batch ~ Uniform(D, batch_size)
```

**Training Update:**
```
θ ← θ - α ∇_θ L(θ)
```

where `α = 1e-3` (learning rate).

## 6. Rule-Based Strategies

### 6.1 Cost Optimized Strategy

```
π_cost(x) = argmin_{s∈S} price_s
```

### 6.2 Reliability Optimized Strategy

```
π_reliability(x) = argmax_{s∈S} reliability_s
```

### 6.3 Hybrid Strategy

```
π_hybrid(x) = {
    EC2_OnDemand, if utilization > 0.8
    EC2_Spot,      if utilization < 0.3
    Lambda,         otherwise
}
```

### 6.4 Workload Aware Strategy

```
π_workload(x) = {
    Lambda,         if CV > 0.5
    EC2_OnDemand,  if mean_demand > 400
    EC2_Spot,       otherwise
}
```

where `CV = std(demand_history) / mean(demand_history)`.

## 7. Optimization Objective

### 7.1 Primary Objective

```
minimize: Σ_{t=0}^T total_cost_t
subject to: latency_t ≤ 200ms (SLA constraint)
```

### 7.2 Multi-Objective Formulation

```
minimize: f(x) = [total_cost, sla_violations, interruptions]
subject to: 
    - latency_t ≤ 200ms
    - instances_{s,t} ≥ 0
    - instances_{s,t} ≤ max_instances_s
```

### 7.3 Reward Engineering

```
R(x, a) = -(cost_weight · total_cost + sla_weight · penalty)
```

where:
- `cost_weight = 1.0`
- `sla_weight = 2.0`

## 8. State Space and Transitions

### 8.1 State Representation

```
S = {s_t | t ∈ T}
```

where each state is:
```
s_t = (demand_t, utilization_t, latency_t, instances_t, prices_t)
```

### 8.2 State Transition Probability

```
P(s_{t+1} | s_t, a_t) = {
    1, if deterministic transition
    P(interruption), if service interruption occurs
}
```

### 8.3 Episode Termination

```
done_t = {
    True, if t ≥ T_max
    False, otherwise
}
```

## 9. Performance Metrics

### 9.1 Cost Metrics

```
total_cost = Σ_{t=0}^T Σ_{s∈S} P_s(instances_{s,t}, demand_{s,t}, 1, t)
cost_per_request = total_cost / Σ_{t=0}^T demand_t
```

### 9.2 SLA Metrics

```
sla_violation_rate = Σ_{t=0}^T I[latency_t > 200] / T
avg_latency = (1/T) Σ_{t=0}^T latency_t
```

### 9.3 Efficiency Metrics

```
resource_efficiency = Σ_{t=0}^T demand_t / Σ_{t=0}^T total_capacity_t
service_utilization_s = (1/T) Σ_{t=0}^T instances_{s,t}
```

## 10. Algorithmic Flow

### 10.1 Training Algorithm

```
1. Initialize Q_θ, target network Q_θ', replay buffer D
2. For episode = 1 to max_episodes:
   a. Initialize state s_0
   b. For t = 0 to T_max:
      i. Select action a_t = ε-greedy(Q_θ(s_t))
      ii. Execute a_t, observe r_t, s_{t+1}
      iii. Store (s_t, a_t, r_t, s_{t+1}) in D
      iv. Sample batch from D
      v. Update Q_θ using batch
      vi. Update target network every C steps
```

### 10.2 Evaluation Algorithm

```
1. Load trained model Q_θ
2. For episode = 1 to n_episodes:
   a. Initialize state s_0
   b. For t = 0 to T_max:
      i. Select action a_t = argmax_a Q_θ(s_t, a)
      ii. Execute a_t, observe r_t, s_{t+1}
      iii. Update metrics
```

## 11. Mathematical Model Summary

### 11.1 System Definition

The Cloud RL Cost Optimization system is formally defined as:

```
M = (S, A, P, R, γ, π)
```

where:
- `S` is the state space (ℝ¹⁰)
- `A` is the action space (12 discrete actions)
- `P` is the transition probability function
- `R` is the reward function
- `γ` is the discount factor (0.99)
- `π` is the policy (RL or rule-based)

### 11.2 Optimization Problem

```
π* = argmax_π E[Σ_{t=0}^T γ^t R(s_t, a_t)]
```

subject to:
- SLA constraints: `latency_t ≤ 200ms`
- Capacity constraints: `0 ≤ instances_{s,t} ≤ max_instances_s`
- Service availability: `P(service_available) ≥ reliability_s`

### 11.3 Key Mathematical Properties

1. **State Space**: Continuous 10-dimensional space
2. **Action Space**: Discrete 12-dimensional space
3. **Reward Function**: Non-linear with penalty terms
4. **Constraints**: Mixed integer-linear programming structure
5. **Objective**: Multi-objective optimization (cost, SLA, reliability)

This mathematical model captures the complete logic, flow, and constraints of the Cloud RL Cost Optimization system in a formal mathematical framework.
