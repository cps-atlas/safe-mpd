# MBD Planner Test Suite

This directory contains comprehensive tests for the Model-Based Diffusion (MBD) planner with support for both default (no demonstration) and demonstration-enabled scenarios.

## Test Structure

### Test Categories

1. **Default Tests**: Tests that rely purely on the reward function without demonstration guidance (default behavior)
2. **Demo Tests**: Tests that use demonstration trajectories to guide the diffusion process (special case)
3. **Utility Tests**: Tests for configuration management and test infrastructure

### Available Test Scenarios

Each scenario has both default (no demo) and demo variants:

- `parking_basic_forward` / `parking_basic_forward_demo`: Basic forward parking
- `parking_basic_backward` / `parking_basic_backward_demo`: Basic backward parking  
- `parking_no_preference` / `parking_no_preference_demo`: No motion preference
- `parking_enforce_forward` / `parking_enforce_forward_demo`: Strict forward enforcement
- `parking_enforce_backward` / `parking_enforce_backward_demo`: Strict backward enforcement

## Running Tests

### Prerequisites

Make sure you're in the MBD source directory:
```bash
cd src/safe_mbd
```

### Run Default Tests (No Demonstration)
```bash
# Default behavior - no flags needed
python tests/test_planners/run_tests.py

# Or explicitly
python tests/test_planners/run_tests.py --default
```

### Run Demo Tests (With Demonstration)
```bash
python tests/test_planners/run_tests.py --demo
```

### Run All Tests
```bash
python tests/test_planners/run_tests.py --all
```

### Run Single Test
```bash
# Default version (no demo)
python tests/test_planners/run_tests.py --single test_parking_basic_forward

# Demo version
python tests/test_planners/run_tests.py --single test_parking_basic_forward_demo
```

### Compare Default vs Demo
```bash
python tests/test_planners/run_tests.py --compare parking_basic_forward
```

### Enable Visualization
```bash
python tests/test_planners/run_tests.py --single test_parking_basic_forward --visualize
```

### List Available Tests
```bash
python tests/test_planners/run_tests.py --list
```

## Test Configuration

### TestConfig Class

The `TestConfig` class extends `MBDConfig` with test-specific fields:

```python
@dataclass
class TestConfig(MBDConfig):
    # Test metadata
    test_name: str = ""
    description: str = ""
    expected_reward_min: float = -1.0
    expected_reward_max: float = 1.0
    timeout_seconds: float = 300
    visualize: bool = False
    enable_demo: bool = False  # NO DEMO BY DEFAULT
    
    # Environment customization
    custom_circular_obstacles: Optional[List] = None
    custom_rectangular_obstacles: Optional[List] = None
    custom_parking_config: Optional[Dict] = None
    
    # Initial/goal positions using geometric parameters
    init_dx: Optional[float] = None
    init_dy: Optional[float] = None
    init_theta1: Optional[float] = None
    init_theta2: Optional[float] = None
    goal_theta1: Optional[float] = None
    goal_theta2: Optional[float] = None
```

### Creating Custom Test Configurations

```python
from fixtures.test_configs import create_demo_variant, create_custom_test_config

# Create demo variant from default scenario
config = create_demo_variant("parking_basic_forward", enable_demo=True)

# Create custom configuration
config = create_custom_test_config(
    "parking_basic_forward",
    Nsample=1000,
    Hsample=30,
    motion_preference=-1,
    visualize=True
)
```

## Environment Setup

### Parking Scenario Configuration

The parking scenario uses a geometric positioning system:

- `init_dx`: Distance from tractor front face to target parking space center (x-direction)
- `init_dy`: Distance from tractor to parking lot entrance line (y-direction)
- `init_theta1`, `init_theta2`: Initial orientations for tractor and trailer
- `goal_theta1`, `goal_theta2`: Goal orientations (positions determined by target parking space)

### Custom Obstacles

```python
config = TestConfig(
    custom_circular_obstacles=[
        [x, y, radius],  # Additional circular obstacles
        [2.0, 3.0, 1.0]
    ],
    custom_rectangular_obstacles=[
        [x, y, width, height, angle],  # Additional rectangular obstacles
        [5.0, 2.0, 2.0, 1.0, 0.0]
    ]
)
```

## Test Results

### Output Files

Test results are saved in `tests/test_planners/results/`:
- `{test_name}_result.json`: Test metadata and timing information
- `{test_name}_arrays.npz`: Action and state trajectories

### Result Validation

Each test validates:
- Reward within expected bounds
- Proper array dimensions
- Goal distance criteria
- Execution within timeout

## Development Guide

### Adding New Test Scenarios

1. **Add to TEST_SCENARIOS** in `fixtures/test_configs.py` (no demo by default):
```python
"new_scenario": TestConfig(
    test_name="new_scenario",
    description="Description of new scenario",
    enable_demo=False,  # Default
    expected_reward_min=0.1,  # Lower expectations without demo
    # ... other configuration parameters
)
```

2. **Add test methods** in `test_mbd_planner.py`:
```python
def test_new_scenario(self):
    """Test new scenario (no demonstration)"""
    self.run_scenario_test("new_scenario")

def test_new_scenario_demo(self):
    """Test new scenario with demonstration"""
    self.run_scenario_test("new_scenario_demo")
```

The demo variant is automatically created by the configuration system.

### Custom Test Environments

```python
from fixtures.test_environments import create_test_tt2d_environment

def test_custom_environment(self):
    config = TestConfig(
        test_name="custom_test",
        enable_demo=False,  # Default
        custom_circular_obstacles=[[0, 0, 2.0]],
        init_dx=-5.0,
        init_dy=3.0
    )
    
    env = create_test_tt2d_environment(config)
    # ... run test
```

## Troubleshooting

### Common Issues

1. **JAX Compilation Warnings**: Normal for first run, cached afterwards
2. **Test Timeout**: Increase `timeout_seconds` in TestConfig
3. **Visualization Not Showing**: Make sure X11 forwarding is enabled if using SSH
4. **Import Errors**: Ensure you're running from the correct directory

### Debug Mode

Enable verbose output:
```bash
python tests/test_planners/run_tests.py --single test_name --visualize
```

### Performance Tips

- Default tests (no demo) are faster and good for development
- Use demo tests to evaluate demonstration effectiveness
- Run single tests during debugging
- Use `--compare` to evaluate default vs demo performance
- Enable visualization only when needed

## Integration with CI/CD

### Quick Test Suite
```bash
# Fast default tests (no demonstration)
python tests/test_planners/run_tests.py

# Demo tests for performance comparison
python tests/test_planners/run_tests.py --demo

# Full test suite
python tests/test_planners/run_tests.py --all
```

### Test Behavior

- **Default behavior** (no flags): Run default tests (no demonstration) - fastest
- **--demo**: Run demo tests (with demonstration) - slower but better performance
- **--all**: Run everything (default + demo + utility) - comprehensive

### Expected Test Times
- Single test: ~30-60 seconds
- Default tests: ~3-5 minutes total (faster without demo)
- Demo tests: ~5-10 minutes total (slower with demo generation)
- All tests: ~8-15 minutes total

## Design Philosophy

The test suite follows the principle that **no demonstration should be the default behavior**:

1. **Baseline Performance**: Default tests establish baseline performance without demonstration
2. **Enhanced Performance**: Demo tests show the benefit of demonstration guidance
3. **Easy Comparison**: Simple comparison between default and demo variants
4. **Development Focus**: Developers typically work with faster default tests
5. **Production Evaluation**: Demo tests validate demonstration effectiveness 