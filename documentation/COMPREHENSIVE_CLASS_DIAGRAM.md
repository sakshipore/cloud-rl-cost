# Cloud RL Cost Optimization - Comprehensive UML Class Diagram

## Project Overview
This document provides a comprehensive UML class diagram for the Cloud RL Cost Optimization project, following proper UML notation with detailed relationships, access modifiers, and complete class specifications.

## UML Class Diagram Specification

```mermaid
classDiagram
    %% Core Environment Classes
    class CloudService {
        -name: String
        -pricing_model: Callable
        -capacity: int
        -startup_time: int
        -reliability: float
        -max_instances: int
        -execution_limit: int
        +__init__(name, pricing_model, capacity, startup_time, reliability, max_instances, execution_limit)
        +calculate_cost(instances, requests, duration, time, **kwargs) float
        +can_handle_workload(demand, instances) bool
        +get_utilization(demand, instances) float
    }

    class EnhancedCloudEnvironment {
        -n_steps: int
        -rng: np.random.Generator
        -workload_type: str
        -services: Dict~str, CloudService~
        -service_instances: Dict~str, int~
        -service_pending: Dict~str, List~int~~
        -latency_target: float
        -sla_penalty: float
        -t: int
        -workload: np.ndarray
        -current_latency: float
        -history: Dict~str, List~
        +__init__(n_steps, seed, workload_type, services)
        +reset(seed) void
        +step(action) Tuple~float, bool, Dict~
        +get_state() np.ndarray
        +get_metrics() Dict~str, Any~
        -_handle_scaling(service_name, scale_action) void
        -_process_pending_instances() void
        -_handle_interruptions() int
        -_calculate_total_capacity() float
        -_calculate_service_demand(service_name, total_demand) int
        -_calculate_latency(utilization) float
        -_update_history(demand, latency, total_cost, sla_violation, service_costs, interruptions) void
    }

    class EnhancedCloudCostGym {
        -env: EnhancedCloudEnvironment
        -n_steps: int
        -workload_type: str
        -service_names: List~str~
        -n_services: int
        -action_space: spaces.Discrete
        -observation_space: spaces.Box
        +__init__(n_steps, seed, workload_type, services)
        +reset(seed, options) Tuple~np.ndarray, Dict~
        +step(action) Tuple~np.ndarray, float, bool, bool, Dict~
        +render(mode) Optional~str~
        +get_metrics() Dict~str, Any~
        +get_history() Dict~str, Any~
        +set_workload_type(workload_type) void
        +get_service_info() Dict~str, Any~
        -_get_info() Dict~str, Any~
        +service_instances: property
        +history: property
        +services: property
        +latency_target: property
    }

    %% Abstract Base Class
    class RuleBasedAgent {
        <<abstract>>
        #name: str
        +__init__(name)
        +predict(observation, info) int*
        +get_strategy_description() str
    }

    %% Concrete Rule-Based Agents
    class CostOptimizedAgent {
        -service_names: List~str~
        +__init__()
        +predict(observation, info) int
    }

    class ReliabilityOptimizedAgent {
        -service_names: List~str~
        -reliability_scores: List~float~
        +__init__()
        +predict(observation, info) int
    }

    class HybridAgent {
        -service_names: List~str~
        -reliability_scores: List~float~
        +__init__()
        +predict(observation, info) int
    }

    class WorkloadAwareAgent {
        -service_names: List~str~
        -demand_history: List~float~
        -history_length: int
        +__init__()
        +predict(observation, info) int
    }

    class ThresholdAgent {
        -service_names: List~str~
        -demand_threshold: int
        -utilization_threshold: float
        +__init__(demand_threshold, utilization_threshold)
        +predict(observation, info) int
    }

    %% Evaluation and Training Classes
    class ComprehensiveEvaluator {
        -output_dir: str
        -metrics: List~str~
        +__init__(output_dir)
        +evaluate_strategy(strategy, env, n_episodes, strategy_name) Dict~str, Any~
        +compare_strategies(strategies, workload_types, n_episodes) Dict~str, Any~
        +generate_comparison_report(comparison_results, save_path) str
        +plot_comparison_results(comparison_results, save_dir) void
        -_calculate_summary_stats(results) Dict~str, Any~
        -_plot_cost_comparison(comparison_results, save_dir) void
        -_plot_sla_comparison(comparison_results, save_dir) void
        -_plot_service_usage(comparison_results, save_dir) void
        -_plot_performance_tradeoffs(comparison_results, save_dir) void
    }

    %% External Dependencies
    class DQN {
        <<external>>
        +learn(total_timesteps, callback) void
        +predict(obs, deterministic) Tuple~int, np.ndarray~
        +save(path) void
    }

    class GymnasiumEnv {
        <<interface>>
        +reset(seed, options) Tuple~np.ndarray, Dict~
        +step(action) Tuple~np.ndarray, float, bool, bool, Dict~
        +render(mode) Optional~str~
    }

    %% Utility Functions (Static)
    class WorkloadGenerator {
        <<utility>>
        +generate_workload(n_steps, seed, workload_type) np.ndarray$
        +generate_workloads(n_steps, types, seed) Dict~str, np.ndarray~$
        +get_workload_characteristics(workload) Dict~str, float~$
        +_generate_diurnal_workload(t, rng) np.ndarray$
        +_generate_steady_workload(t, rng) np.ndarray$
        +_generate_batch_workload(t, rng) np.ndarray$
        +_generate_bursty_workload(t, rng) np.ndarray$
    }

    class ServiceFactory {
        <<utility>>
        +create_service(service_type, **kwargs) CloudService$
        +get_all_services() Dict~str, CloudService~$
        +on_demand_pricing(instances, requests, duration, time, base_price, **kwargs) float$
        +spot_pricing(instances, requests, duration, time, base_price, **kwargs) float$
        +serverless_pricing(instances, requests, duration, time, base_price, **kwargs) float$
        +container_pricing(instances, requests, duration, time, base_price, **kwargs) float$
    }

    class AgentFactory {
        <<utility>>
        +create_agent(agent_type, **kwargs) RuleBasedAgent$
    }

    %% Relationships - Inheritance
    RuleBasedAgent <|-- CostOptimizedAgent : extends
    RuleBasedAgent <|-- ReliabilityOptimizedAgent : extends
    RuleBasedAgent <|-- HybridAgent : extends
    RuleBasedAgent <|-- WorkloadAwareAgent : extends
    RuleBasedAgent <|-- ThresholdAgent : extends

    %% Relationships - Implementation
    EnhancedCloudCostGym ..|> GymnasiumEnv : implements

    %% Relationships - Composition (Strong Ownership)
    EnhancedCloudEnvironment *-- CloudService : contains 1..*
    EnhancedCloudCostGym *-- EnhancedCloudEnvironment : contains 1

    %% Relationships - Aggregation (Weak Ownership)
    EnhancedCloudEnvironment o-- WorkloadGenerator : uses
    ComprehensiveEvaluator o-- EnhancedCloudCostGym : evaluates
    ComprehensiveEvaluator o-- RuleBasedAgent : compares

    %% Relationships - Association (Usage)
    EnhancedCloudEnvironment --> CloudService : uses
    EnhancedCloudCostGym --> EnhancedCloudEnvironment : wraps
    ComprehensiveEvaluator --> EnhancedCloudCostGym : evaluates
    ComprehensiveEvaluator --> RuleBasedAgent : compares
    AgentFactory --> RuleBasedAgent : creates
    ServiceFactory --> CloudService : creates

    %% Relationships - Dependency (Temporary Usage)
    EnhancedCloudEnvironment ..> WorkloadGenerator : depends on
    EnhancedCloudEnvironment ..> ServiceFactory : depends on
    CostOptimizedAgent ..> CloudService : depends on
    ReliabilityOptimizedAgent ..> CloudService : depends on
    HybridAgent ..> CloudService : depends on
    WorkloadAwareAgent ..> CloudService : depends on
    ThresholdAgent ..> CloudService : depends on

    %% Styling
    classDef environmentClass fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef rlClass fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef baselineClass fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef utilityClass fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef externalClass fill:#f5f5f5,stroke:#616161,stroke-width:1px,stroke-dasharray: 5 5
    classDef interfaceClass fill:#e8eaf6,stroke:#3f51b5,stroke-width:2px,stroke-dasharray: 5 5

    class CloudService,EnhancedCloudEnvironment,EnhancedCloudCostGym environmentClass
    class ComprehensiveEvaluator rlClass
    class RuleBasedAgent,CostOptimizedAgent,ReliabilityOptimizedAgent,HybridAgent,WorkloadAwareAgent,ThresholdAgent,AgentFactory baselineClass
    class WorkloadGenerator,ServiceFactory utilityClass
    class DQN externalClass
    class GymnasiumEnv interfaceClass
```

## Detailed Class Specifications

### Core Environment Classes

#### CloudService
- **Purpose**: Represents individual cloud service types with specific characteristics
- **Key Attributes**:
  - `name` (private): Service name (e.g., "EC2 On-Demand")
  - `pricing_model` (private): Function for cost calculation
  - `capacity` (private): Request handling capacity per instance
  - `startup_time` (private): Minutes required to start up
  - `reliability` (private): Probability of not being interrupted (0-1)
  - `max_instances` (private): Maximum instances allowed
  - `execution_limit` (private): Maximum execution time (for serverless)
- **Key Methods**:
  - `calculate_cost()` (public): Calculate cost based on usage parameters
  - `can_handle_workload()` (public): Check if service can handle demand
  - `get_utilization()` (public): Calculate utilization percentage

#### EnhancedCloudEnvironment
- **Purpose**: Core simulation environment that models cloud services with different characteristics
- **Key Attributes**:
  - `services` (private): Dictionary of available cloud services
  - `service_instances` (private): Current number of instances per service
  - `service_pending` (private): Pending instances waiting to start
  - `workload` (private): Generated workload pattern
  - `history` (private): Metrics tracking (demand, latency, costs, etc.)
- **Key Methods**:
  - `reset()` (public): Initialize environment state
  - `step()` (public): Execute one simulation step
  - `get_state()` (public): Return current state as numpy array
  - `get_metrics()` (public): Calculate comprehensive metrics

#### EnhancedCloudCostGym
- **Purpose**: OpenAI Gym-compatible wrapper for the environment
- **Key Attributes**:
  - `env` (private): Underlying EnhancedCloudEnvironment
  - `action_space` (private): Discrete action space
  - `observation_space` (private): State space with demand, utilization, latency
- **Key Methods**:
  - `reset()` (public): Reset environment and return initial observation
  - `step()` (public): Take action and return (observation, reward, done, info)
  - `render()` (public): Render the environment

### Rule-Based Agent Classes

#### RuleBasedAgent (Abstract Base Class)
- **Purpose**: Abstract base class for all rule-based strategies
- **Key Attributes**:
  - `name` (protected): Strategy name
- **Key Methods**:
  - `predict()` (public, abstract): Abstract method for action prediction
  - `get_strategy_description()` (public): Return strategy description

#### Concrete Rule-Based Agents
1. **CostOptimizedAgent**: Always chooses cheapest service
2. **ReliabilityOptimizedAgent**: Prioritizes most reliable service
3. **HybridAgent**: Uses different strategies based on conditions
4. **WorkloadAwareAgent**: Adapts to workload patterns
5. **ThresholdAgent**: Uses simple thresholds for decisions

### Evaluation and Training Classes

#### ComprehensiveEvaluator
- **Purpose**: Comprehensive evaluation framework for comparing strategies
- **Key Attributes**:
  - `output_dir` (private): Directory for saving results
  - `metrics` (private): List of metrics to track
- **Key Methods**:
  - `evaluate_strategy()` (public): Evaluate single strategy
  - `compare_strategies()` (public): Compare multiple strategies
  - `generate_comparison_report()` (public): Generate text report
  - `plot_comparison_results()` (public): Generate visualization plots

### Utility Classes

#### WorkloadGenerator (Static Utility)
- **Purpose**: Generate synthetic workload patterns for testing
- **Key Methods** (all static):
  - `generate_workload()`: Main function to generate workloads
  - `generate_workloads()`: Generate multiple workload types
  - `get_workload_characteristics()`: Analyze workload properties

#### ServiceFactory (Static Utility)
- **Purpose**: Factory for creating cloud service instances
- **Key Methods** (all static):
  - `create_service()`: Create service from predefined configurations
  - `get_all_services()`: Get all predefined service types
  - Pricing model functions: `on_demand_pricing()`, `spot_pricing()`, etc.

#### AgentFactory (Static Utility)
- **Purpose**: Factory for creating rule-based agents
- **Key Methods** (all static):
  - `create_agent()`: Create agent of specified type

## Relationship Explanations

### Inheritance Relationships
- **RuleBasedAgent <|-- CostOptimizedAgent**: CostOptimizedAgent extends RuleBasedAgent because it implements the abstract predict() method with cost-optimized logic
- **RuleBasedAgent <|-- ReliabilityOptimizedAgent**: ReliabilityOptimizedAgent extends RuleBasedAgent because it implements the abstract predict() method with reliability-optimized logic
- **RuleBasedAgent <|-- HybridAgent**: HybridAgent extends RuleBasedAgent because it implements the abstract predict() method with hybrid logic
- **RuleBasedAgent <|-- WorkloadAwareAgent**: WorkloadAwareAgent extends RuleBasedAgent because it implements the abstract predict() method with workload-aware logic
- **RuleBasedAgent <|-- ThresholdAgent**: ThresholdAgent extends RuleBasedAgent because it implements the abstract predict() method with threshold-based logic

### Implementation Relationships
- **EnhancedCloudCostGym ..|> GymnasiumEnv**: EnhancedCloudCostGym implements GymnasiumEnv interface because it provides the required reset(), step(), and render() methods for Gymnasium compatibility

### Composition Relationships (Strong Ownership)
- **EnhancedCloudEnvironment *-- CloudService**: EnhancedCloudEnvironment has strong ownership of CloudService objects because the environment manages the lifecycle of service instances and cannot exist without them
- **EnhancedCloudCostGym *-- EnhancedCloudEnvironment**: EnhancedCloudCostGym has strong ownership of EnhancedCloudEnvironment because the gym wrapper cannot exist without the underlying environment

### Aggregation Relationships (Weak Ownership)
- **EnhancedCloudEnvironment o-- WorkloadGenerator**: EnhancedCloudEnvironment uses WorkloadGenerator but doesn't own it because workload generation is a utility function
- **ComprehensiveEvaluator o-- EnhancedCloudCostGym**: ComprehensiveEvaluator uses EnhancedCloudCostGym for evaluation but doesn't own it
- **ComprehensiveEvaluator o-- RuleBasedAgent**: ComprehensiveEvaluator uses RuleBasedAgent for comparison but doesn't own it

### Association Relationships (Usage)
- **EnhancedCloudEnvironment --> CloudService**: EnhancedCloudEnvironment uses CloudService objects to calculate costs and handle workloads
- **EnhancedCloudCostGym --> EnhancedCloudEnvironment**: EnhancedCloudCostGym uses EnhancedCloudEnvironment to get state and execute actions
- **ComprehensiveEvaluator --> EnhancedCloudCostGym**: ComprehensiveEvaluator uses EnhancedCloudCostGym to evaluate strategies
- **ComprehensiveEvaluator --> RuleBasedAgent**: ComprehensiveEvaluator uses RuleBasedAgent to compare with RL strategies
- **AgentFactory --> RuleBasedAgent**: AgentFactory creates RuleBasedAgent instances
- **ServiceFactory --> CloudService**: ServiceFactory creates CloudService instances

### Dependency Relationships (Temporary Usage)
- **EnhancedCloudEnvironment ..> WorkloadGenerator**: EnhancedCloudEnvironment depends on WorkloadGenerator because it calls generate_workload() function
- **EnhancedCloudEnvironment ..> ServiceFactory**: EnhancedCloudEnvironment depends on ServiceFactory because it calls get_all_services() function
- **Rule-Based Agents ..> CloudService**: All rule-based agents depend on CloudService because they need to understand service characteristics for decision making

## Multiplicity and Cardinality

- **EnhancedCloudEnvironment *-- CloudService (1..*)**: One environment contains multiple cloud services
- **EnhancedCloudCostGym *-- EnhancedCloudEnvironment (1)**: One gym wrapper contains exactly one environment
- **ComprehensiveEvaluator o-- EnhancedCloudCostGym (1..*)**: One evaluator can evaluate multiple environments
- **ComprehensiveEvaluator o-- RuleBasedAgent (1..*)**: One evaluator can compare multiple rule-based agents

## Static and Abstract Members

- **Static Members**: All methods in WorkloadGenerator, ServiceFactory, and AgentFactory are static (marked with $)
- **Abstract Members**: RuleBasedAgent.predict() method is abstract (marked with *)
- **Access Modifiers**: 
  - `+` for public members
  - `-` for private members  
  - `#` for protected members
  - `$` for static members
  - `*` for abstract members

This comprehensive UML class diagram captures all aspects of the Cloud RL Cost Optimization project with proper UML notation, complete relationships, and detailed explanations of why each relationship exists.
