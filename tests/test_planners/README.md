# MBD Planner Tests

This directory contains comprehensive tests for the Model-Based Diffusion (MBD) planner functionality.

## Directory Structure

```
tests/test_planners/
├── __init__.py                     # Module initialization
├── README.md                       # This file
├── run_tests.py                    # Test runner script
├── test_base.py                    # Base test class
├── test_mbd_planner.py            # Main integration tests
├── fixtures/                       # Test configurations and utilities
│   ├── __init__.py
│   ├── test_configs.py             # TestConfig dataclass and scenarios
│   └── test_environments.py        # Environment setup utilities
└── results/                        # Test results (created during test runs)
    ├── {test_name}_result.json     # JSON result files
    └── {test_name}_arrays.npz      # Numpy arrays (actions, states)
```

## Running Tests

### Quick Start

```bash
# Run all tests
python -m unittest tests.test_planners

# Run specific test module
python -m unittest tests.test_planners.test_mbd_planner

# Run single test
python -m unittest tests.test_planners.test_mbd_planner.TestMBDPlanner.test_basic_forward_parking
```

### Using the Test Runner Script

```bash
cd tests/test_planners

# Run all tests
python run_tests.py

# Run fast subset (smaller configurations)
python run_tests.py --fast

# Run single test
python run_tests.py --single test_basic_forward_parking

# List available tests
python run_tests.py --list
```

## Test Categories

### Integration Tests (`test_mbd_planner.py`)

- **test_basic_forward_parking**: Basic forward parking in case2 scenario
- **test_basic_backward_parking**: Basic backward parking in case2 scenario  
- **test_no_motion_preference**: Parking with no motion preference
- **test_demo_enabled_vs_disabled**: Compare results with/without demonstration
- **test_jit_compilation_timing**: Test JIT compilation behavior and caching
- **test_different_sample_sizes**: Test diffusion with different sample sizes

## Test Configuration

Tests use the `TestConfig` dataclass defined in `fixtures/test_configs.py`. Key parameters:

```python
@dataclass
class TestConfig:
    # Test metadata
    test_name: str
    description: str
    expected_reward_min: float
    expected_reward_max: float
    
    # MBD parameters (optimized for testing)
    seed: int = 42                  # Fixed for reproducibility
    Nsample: int = 1000            # Smaller than production
    Hsample: int = 25              # Shorter horizon
    Ndiffuse: int = 50             # Fewer diffusion steps
    enable_demo: bool = True
    motion_preference: int = 0     # 0=none, ±1=preference, ±2=enforce
    
    # Environment customization
    custom_init_pos: Optional[Tuple] = None
    custom_goal_pos: Optional[Tuple] = None
    custom_circular_obstacles: Optional[List] = None
    custom_rectangular_obstacles: Optional[List] = None
    
    # Validation criteria
    min_final_distance_to_goal: float = 2.0
    max_hitch_angle_violation: float = 0.1
```

### Predefined Scenarios

Available in `TEST_SCENARIOS` dictionary:

- `case2_basic_forward`: Forward parking scenario
- `case2_basic_backward`: Backward parking scenario
- `case2_no_preference`: No motion preference scenario

## Test Results and Reproducibility

### Result Files

Each test run generates:

1. **JSON Result File** (`{test_name}_result.json`):
   ```json
   {
     "config": {...},
     "reward": 0.7234,
     "final_state": [x, y, theta1, theta2],
     "timing": {...},
     "timestamp": "2025-01-01T12:00:00",
     "git_commit": "abc123def",
     "test_summary": {
       "goal_distance": 1.23,
       "max_hitch_violation": 0.05,
       "trajectory_length": 26,
       "action_sequence_length": 25
     }
   }
   ```

2. **Numpy Arrays** (`{test_name}_arrays.npz`):
   - `actions`: Action sequence array (Hsample, 2)
   - `states`: State trajectory array (Hsample+1, 4)

### Reproducibility

Tests are designed for reproducibility:

- **Fixed Seeds**: All tests use `seed=42` by default
- **Git Commit Tracking**: Results include current git commit hash
- **Configuration Serialization**: Complete test configuration saved
- **Deterministic Parameters**: Consistent MBD and environment parameters

## Reproducing Results for Presentations

### Command Format for Slides

For each result in presentation slides:

```bash
# Example: Case2 Forward Parking Test
git checkout abc123def
cd tests/test_planners
python run_tests.py --single test_basic_forward_parking
```

### Expected Results

Each test has defined expected ranges:

- **Forward Parking**: Reward 0.4-1.0, Goal distance < 2.0m
- **Backward Parking**: Reward 0.3-1.0, Goal distance < 2.0m  
- **No Preference**: Reward 0.3-1.0, Goal distance < 2.0m

## Adding New Tests

### 1. Define Test Configuration

Add to `fixtures/test_configs.py`:

```python
"new_scenario": TestConfig(
    test_name="new_scenario",
    description="Description of test scenario",
    motion_preference=1,
    expected_reward_min=0.3,
    expected_reward_max=0.8,
    custom_init_pos=(-5, -3, 0, 0),
    custom_goal_pos=(5, 5, np.pi/2, np.pi/2)
)
```

### 2. Create Test Method

Add to `test_mbd_planner.py`:

```python
def test_new_scenario(self):
    """Test description"""
    config = get_test_config("new_scenario")
    reward, actions, states, timing = self.run_mbd_test(config)
    
    # Additional specific validations
    self.assertGreater(reward, 0.3, "Custom validation message")
```

### 3. Custom Environment Setup

Modify `fixtures/test_environments.py` if needed for special environment configurations.

## Test Performance

Typical test execution times:

- **Single basic test**: ~15-30 seconds
- **Fast test suite**: ~2-3 minutes
- **Full test suite**: ~10-15 minutes

Tests use smaller configurations than production (1000 samples vs 20000, 25 horizon vs 50, 50 diffusion steps vs 150) for faster execution while maintaining test validity.

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure MBD source path is correctly added
2. **JAX Compilation**: First test run includes JIT compilation overhead
3. **Memory Issues**: Reduce `Nsample`, `Hsample`, or `Ndiffuse` for resource-constrained systems
4. **Timeout Errors**: Increase `timeout_seconds` in test configuration

### Debug Mode

For detailed output:

```bash
python -m unittest tests.test_planners.test_mbd_planner -v
```

### Clearing Results

```bash
rm -rf tests/test_planners/results/*
``` 