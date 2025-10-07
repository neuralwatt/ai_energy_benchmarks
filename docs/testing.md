# Testing Guide

This guide explains how to run and write tests for the AI Energy Benchmarks framework.

## Test Structure

```
tests/
├── unit/                  # Unit tests for individual components
│   ├── test_vllm_backend.py
│   ├── test_dataset.py
│   ├── test_config.py
│   └── test_csv_reporter.py
└── integration/           # Integration tests for full workflows
    └── test_benchmark_runner.py
```

## Running Tests

### All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=ai_energy_benchmarks --cov-report=html
```

### Specific Test Categories

```bash
# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run specific test file
pytest tests/unit/test_vllm_backend.py

# Run specific test
pytest tests/unit/test_vllm_backend.py::TestVLLMBackend::test_initialization
```

## Test Coverage

View coverage report:

```bash
# Generate HTML coverage report
pytest --cov=ai_energy_benchmarks --cov-report=html

# Open report
open htmlcov/index.html  # On macOS
xdg-open htmlcov/index.html  # On Linux
```

Current coverage targets:
- Unit tests: >80% coverage
- Integration tests: Key workflows covered
- End-to-end tests: Full benchmark execution

## Writing Tests

### Unit Test Example

```python
# tests/unit/test_my_component.py
import pytest
from ai_energy_benchmarks.my_module import MyComponent

class TestMyComponent:
    """Test MyComponent."""

    def test_initialization(self):
        """Test component initialization."""
        component = MyComponent(param="value")
        assert component.param == "value"

    def test_method_success(self):
        """Test successful method execution."""
        component = MyComponent()
        result = component.my_method()
        assert result is True

    def test_method_failure(self):
        """Test method failure handling."""
        component = MyComponent()
        with pytest.raises(ValueError):
            component.my_method(invalid_param=True)
```

### Mocking External Dependencies

```python
from unittest.mock import Mock, patch

@patch('ai_energy_benchmarks.backends.vllm.requests.get')
def test_with_mock(mock_get):
    """Test with mocked HTTP request."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'data': []}
    mock_get.return_value = mock_response

    # Your test code here
    backend = VLLMBackend(endpoint="http://test", model="test")
    result = backend.validate_environment()
    assert result is True
```

### Integration Test Example

```python
# tests/integration/test_workflow.py
import pytest
from ai_energy_benchmarks.runner import BenchmarkRunner
from ai_energy_benchmarks.config.parser import BenchmarkConfig

class TestWorkflow:
    """Integration tests for workflows."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = BenchmarkConfig()
        config.scenario.num_samples = 2
        return config

    def test_full_workflow(self, config):
        """Test complete benchmark workflow."""
        runner = BenchmarkRunner(config)
        results = runner.run()
        assert results['summary']['total_prompts'] == 2
```

## Test Fixtures

### Common Fixtures

```python
@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return str(output_dir)

@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = BenchmarkConfig()
    config.name = "test"
    config.scenario.num_samples = 5
    return config
```

## Continuous Integration

Tests run automatically on:
- Push to main branch
- Pull requests
- Release tags

See `.github/workflows/` for CI configuration.

## Debugging Tests

### Run with Debug Output

```bash
# Show print statements
pytest -s

# Show full traceback
pytest --tb=long

# Drop into debugger on failure
pytest --pdb
```

### Debug Specific Test

```python
# In test file, add breakpoint
def test_debug_example():
    result = some_function()
    import pdb; pdb.set_trace()  # Debugger starts here
    assert result == expected
```

## Best Practices

1. **One Assertion Per Test**: Each test should verify one behavior
2. **Clear Test Names**: Use descriptive names like `test_method_with_invalid_input_raises_error`
3. **Mock External Deps**: Mock HTTP requests, file I/O, GPU calls
4. **Use Fixtures**: Reuse common setup code
5. **Test Edge Cases**: Test boundary conditions and error paths
6. **Fast Tests**: Keep unit tests fast (<1s each)

## Common Issues

### Import Errors

```bash
# Install package in editable mode
pip install -e .
```

### Missing Dependencies

```bash
# Install test dependencies
pip install -e ".[dev]"
```

### Failing Tests After Changes

```bash
# Clear pytest cache
pytest --cache-clear

# Regenerate test data
rm -rf .pytest_cache
```

## See Also

- [Contributing Guide](./contributing.md)
- [Development Setup](./development.md)
- [CI/CD Configuration](../.github/workflows/)
