# Reward Function Design Documentation

## Overview

The reward function is a critical component of the reinforcement learning system, as it directly guides the agent's learning process. This document describes the design rationale, components, and parameter tuning guidelines for the reward function used in the DQN-based adaptive service selection system.

## Reward Function Components

The reward function balances multiple objectives:

```
R(s, a) = cost_term + latency_penalty + availability_penalty + high_load_bonus
```

Where:
- `cost_term = -total_cost` (negative because we minimize cost)
- `latency_penalty = -sla_penalty * I(latency > latency_target)`
- `availability_penalty = -availability_penalty * I(availability < availability_target)`
- `high_load_bonus = +high_load_bonus * I(utilization >= threshold AND SLA_met)`

### 1. Cost Term

**Component**: `cost_term = -total_cost`

**Rationale**: 
- Primary objective is cost minimization
- Negative sign ensures higher reward for lower costs
- Directly optimizes the main business objective

**Design Choice**: 
- Uses total cost (sum of all service costs) rather than per-service cost
- Encourages overall cost reduction across all services

### 2. Latency Penalty

**Component**: `latency_penalty = -sla_penalty * I(latency > latency_target)`

**Rationale**:
- Ensures SLA compliance for latency requirements
- Penalty is applied only when latency exceeds target (200ms)
- Binary indicator ensures clear signal to the agent

**Parameters**:
- `sla_penalty = 2.0` (default)
- `latency_target = 200` ms

**Design Choice**:
- Binary penalty (0 or -2.0) provides clear signal
- Penalty magnitude (2.0) balances cost and SLA objectives
- Higher penalty values prioritize SLA compliance more

### 3. Availability Penalty

**Component**: `availability_penalty = -availability_penalty * I(availability < availability_target)`

**Rationale**:
- Ensures service availability meets targets
- Availability calculated as `1 - interruption_rate` per service
- Penalty applied when any service falls below target availability

**Parameters**:
- `availability_penalty = 1.5` (default)
- Service-specific `availability_target` (typically 0.99 for 99%)

**Design Choice**:
- Slightly lower penalty than latency (1.5 vs 2.0) reflects that availability violations are less critical than latency violations
- Per-service availability tracking allows fine-grained control

### 4. High Load Bonus

**Component**: `high_load_bonus = +high_load_bonus * I(utilization >= threshold AND SLA_met)`

**Rationale**:
- Rewards agent for maintaining SLA compliance under high load
- Encourages efficient resource utilization
- Provides positive reinforcement for good performance

**Parameters**:
- `high_load_bonus = 0.5` (default)
- `high_load_threshold = 0.8` (80% utilization)

**Design Choice**:
- Positive reward provides incentive for good performance
- Only awarded when both high load AND SLA compliance are met
- Smaller magnitude (0.5) compared to penalties ensures cost remains primary objective

## Design Rationale

### Multi-Objective Optimization

The reward function addresses a multi-objective optimization problem:

1. **Primary Objective**: Minimize cost
2. **Constraint 1**: Maintain latency SLA (latency ≤ 200ms)
3. **Constraint 2**: Maintain availability SLA (availability ≥ target)
4. **Secondary Objective**: Efficient resource utilization

### Balancing Trade-offs

The relative magnitudes of reward components determine the trade-off between objectives:

- **Cost vs SLA**: `sla_penalty = 2.0` means that violating SLA is equivalent to spending $2.00 more
- **Latency vs Availability**: Latency penalty (2.0) is higher than availability penalty (1.5), reflecting that latency violations are more critical
- **Cost vs Efficiency**: High load bonus (0.5) is small relative to cost, ensuring cost remains primary

### Academic Justification

This reward function design is inspired by:

1. **Multi-objective RL**: Balances competing objectives through weighted components
2. **Constraint satisfaction**: Uses penalties to enforce SLA constraints
3. **Incentive alignment**: Positive rewards for desired behaviors (high-load SLA compliance)
4. **Sparse rewards**: Binary indicators provide clear learning signals

## Parameter Tuning Guidelines

### Tuning `sla_penalty`

**Effect**: Controls trade-off between cost and latency SLA compliance

**Guidelines**:
- **Low values (0.5-1.0)**: Agent prioritizes cost, may violate SLA more frequently
- **Medium values (1.5-2.5)**: Balanced trade-off (recommended)
- **High values (3.0+)**: Agent prioritizes SLA, may overspend on resources

**Recommendation**: Start with 2.0, adjust based on observed SLA violation rates

### Tuning `availability_penalty`

**Effect**: Controls trade-off between cost and availability SLA compliance

**Guidelines**:
- **Low values (0.5-1.0)**: Agent may tolerate more interruptions
- **Medium values (1.0-2.0)**: Balanced trade-off (recommended)
- **High values (2.5+)**: Agent avoids interruption-prone services (e.g., Spot instances)

**Recommendation**: Start with 1.5, adjust based on availability requirements

### Tuning `high_load_bonus`

**Effect**: Encourages efficient resource utilization under high load

**Guidelines**:
- **Low values (0.1-0.3)**: Minimal incentive
- **Medium values (0.4-0.6)**: Moderate incentive (recommended)
- **High values (0.7+)**: May cause agent to artificially increase load

**Recommendation**: Start with 0.5, adjust based on utilization patterns

### Tuning `high_load_threshold`

**Effect**: Defines what constitutes "high load"

**Guidelines**:
- **Low values (0.6-0.7)**: Bonus awarded more frequently
- **Medium values (0.75-0.85)**: Balanced (recommended)
- **High values (0.9+)**: Bonus awarded rarely

**Recommendation**: Start with 0.8 (80% utilization)

## Reward Function Analysis

### Reward Component Tracking

The system tracks all reward components separately for analysis:

- `cost_term`: Cumulative cost impact
- `latency_penalty`: Cumulative latency penalty
- `availability_penalty`: Cumulative availability penalty
- `high_load_bonus`: Cumulative bonus received

This allows for:
- Understanding which components dominate the reward
- Identifying if penalties are too high/low
- Analyzing reward distribution over time

### Visualization

The `rl/reward_analysis.py` module provides visualizations:

1. **Reward vs Time**: Shows reward evolution during training/evaluation
2. **Reward vs Workload Intensity**: Shows how reward varies with demand
3. **Reward Component Breakdown**: Stacked area chart showing component contributions
4. **Reward Component Comparison**: Bar chart comparing total component values

## Comparison with VpQ-Inspired Baseline

The VpQ-inspired baseline uses a **cost-only** reward function:

```
R_vpq(s, a) = -total_cost
```

This serves as a comparison point to demonstrate:
- Impact of SLA awareness on cost optimization
- Trade-off between cost and SLA compliance
- Value of multi-objective reward design

## Best Practices

1. **Start with default parameters**: Default values are tuned for balanced performance
2. **Monitor reward components**: Use visualization tools to understand reward distribution
3. **Adjust incrementally**: Make small changes (0.5 increments) and observe effects
4. **Consider workload characteristics**: Different workloads may require different parameter values
5. **Validate on multiple workloads**: Ensure parameters work across different workload types

## References

- Multi-objective reinforcement learning literature
- Constraint satisfaction in RL
- Reward shaping techniques
- VpQ-learning for pricing optimization (inspiration for baseline)

