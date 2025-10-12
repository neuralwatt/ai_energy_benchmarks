# AI Energy Benchmarks - Pip Package Migration Plan

## Executive Summary

This document outlines the migration of `ai_energy_benchmarks` from a directory-copy dependency to a proper pip-installable package. The package is already configured with `pyproject.toml` and `setup.py`, but is currently integrated into AIEnergyScore via Docker `COPY` commands. This migration will simplify builds, enable version management, and improve reusability.

## Current State

### Dependency Pattern
- **AIEnergyScore**: Copies entire `ai_energy_benchmarks/` directory into Docker image
  ```dockerfile
  COPY ai_energy_benchmarks /ai_energy_benchmarks
  RUN pip install -e /ai_energy_benchmarks
  ```
- **Build Context**: Requires parent directory (`~/src/`) to access both projects
- **neuralwatt_cloud**: No current dependency (potential future library usage)

### Issues with Current Approach
1. Docker builds must run from parent directory
2. No version management or pinning
3. Entire source tree copied (including .git, tests, docs)
4. Cannot independently version AIEnergyScore and ai_energy_benchmarks
5. Difficult to distribute to partners or external users

## Migration Phases

---

## Phase 1: Internal Wheel Distribution

**Goal**: Convert to wheel-based installation without external dependencies or publication.

**Timeline**: 1-2 days

**Risk Level**: Low

### 1.1 Build Infrastructure

#### Create Build Script

**File**: `ai_energy_benchmarks/build_wheel.sh`

```bash
#!/bin/bash
# Build wheel for ai_energy_benchmarks package

set -e

echo "Building ai_energy_benchmarks wheel..."

# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build wheel and source distribution
python -m build

echo "✓ Wheel built successfully"
echo ""
echo "Wheel location: dist/ai_energy_benchmarks-$(cat VERSION.txt)-py3-none-any.whl"
echo ""
echo "Install with:"
echo "  pip install dist/ai_energy_benchmarks-*.whl"
echo ""
echo "Or install with extras:"
echo "  pip install 'dist/ai_energy_benchmarks-*.whl[pytorch]'"
echo "  pip install 'dist/ai_energy_benchmarks-*.whl[all]'"
```

**Installation**:
```bash
cd ai_energy_benchmarks
chmod +x build_wheel.sh
```

#### Verify Build Dependencies

Add to `ai_energy_benchmarks/pyproject.toml` if not present:

```toml
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"
```

### 1.2 Update AIEnergyScore Integration

#### Modify Dockerfile

**File**: `AIEnergyScore/Dockerfile`

**Before**:
```dockerfile
# Install ai_energy_benchmarks (optional backend)
# Copy and install from local source
COPY ai_energy_benchmarks /ai_energy_benchmarks
RUN pip install -e /ai_energy_benchmarks
```

**After**:
```dockerfile
# Install ai_energy_benchmarks (optional backend)
# Copy and install from wheel
COPY ai_energy_benchmarks/dist/ai_energy_benchmarks-*.whl /tmp/
RUN pip install /tmp/ai_energy_benchmarks-*.whl[pytorch]
```

#### Update Build Script

**File**: `AIEnergyScore/build.sh`

**Before**:
```bash
#!/bin/bash
# Build AIEnergyScore Docker image with ai_energy_benchmarks support

set -e

echo "Building AIEnergyScore Docker image with ai_energy_benchmarks support..."
echo "Build context: ~/src/"
echo ""

cd ~/src/

docker build \
    -f AIEnergyScore/Dockerfile \
    -t ai_energy_score \
    .
```

**After**:
```bash
#!/bin/bash
# Build AIEnergyScore Docker image with ai_energy_benchmarks support

set -e

echo "Step 1: Building ai_energy_benchmarks wheel..."
cd ~/src/ai_energy_benchmarks
./build_wheel.sh

echo ""
echo "Step 2: Building AIEnergyScore Docker image..."
cd ~/src/

docker build \
    -f AIEnergyScore/Dockerfile \
    -t ai_energy_score \
    .

echo ""
echo "✓ Docker image 'ai_energy_score' built successfully"
echo ""
echo "Usage examples:"
echo "  # Default (optimum-benchmark):"
echo "  docker run --gpus all ai_energy_score --config-name text_generation"
echo ""
echo "  # PyTorch backend (ai_energy_benchmarks):"
echo "  docker run --gpus all -e BENCHMARK_BACKEND=pytorch ai_energy_score --config-name text_generation"
echo ""
echo "  # vLLM backend (ai_energy_benchmarks):"
echo "  docker run --gpus all -e BENCHMARK_BACKEND=vllm -e VLLM_ENDPOINT=http://host.docker.internal:8000/v1 ai_energy_score --config-name text_generation"
```

### 1.3 Add .dockerignore

**File**: `ai_energy_benchmarks/.dockerignore`

```
# When using ai_energy_benchmarks in other Docker builds,
# only copy the dist/ directory

*
!dist/
!dist/*.whl
```

### 1.4 Update Documentation

**File**: `ai_energy_benchmarks/README.md`

Add section after "Installation":

```markdown
### Building for Distribution

To create a wheel for use in other projects:

```bash
# Build wheel
./build_wheel.sh

# Install from wheel
pip install dist/ai_energy_benchmarks-*.whl

# Install with optional dependencies
pip install 'dist/ai_energy_benchmarks-*.whl[pytorch]'
pip install 'dist/ai_energy_benchmarks-*.whl[all]'
```

The wheel can be copied into Docker images or shared with other projects without requiring the full source tree.
```

### 1.5 Testing Phase 1

**Test Checklist**:

1. **Build wheel locally**:
   ```bash
   cd ai_energy_benchmarks
   ./build_wheel.sh
   ls -lh dist/
   ```

2. **Verify wheel contents**:
   ```bash
   unzip -l dist/ai_energy_benchmarks-*.whl
   # Should contain only package code, not tests/docs/git
   ```

3. **Test local installation**:
   ```bash
   python -m venv test_env
   source test_env/bin/activate
   pip install dist/ai_energy_benchmarks-*.whl[pytorch]
   python -c "from ai_energy_benchmarks.runner import BenchmarkRunner; print('✓ Import successful')"
   ai-energy-benchmark --help
   ```

4. **Build AIEnergyScore with new method**:
   ```bash
   cd ~/src/AIEnergyScore
   ./build.sh
   ```

5. **Verify AIEnergyScore functionality**:
   ```bash
   # Test PyTorch backend
   docker run --gpus all -e BENCHMARK_BACKEND=pytorch ai_energy_score \
     --config-name text_generation

   # Test vLLM backend
   # (Start vLLM server first)
   docker run --gpus all -e BENCHMARK_BACKEND=vllm \
     -e VLLM_ENDPOINT=http://host.docker.internal:8000/v1 \
     ai_energy_score --config-name text_generation
   ```

### 1.6 Benefits of Phase 1

- ✅ Smaller Docker build context (only wheel, not full source)
- ✅ Build can run from any directory (no parent context needed)
- ✅ Cleaner separation: build wheel once, use many times
- ✅ Faster Docker builds (no copying unnecessary files)
- ✅ Foundation for Phase 3 (public distribution)
- ✅ No external dependencies or infrastructure required

### 1.7 Rollback Plan

If Phase 1 encounters issues:

1. Revert `AIEnergyScore/Dockerfile` to COPY method
2. Revert `AIEnergyScore/build.sh` to original
3. Document issues in `ai_helpers/pip_package_migration_issues.md`
4. Previous Docker images remain functional

---

## Phase 3: Public PyPI Distribution

**Goal**: Publish `ai_energy_benchmarks` to PyPI for public distribution.

**Timeline**: 1 week (includes testing and documentation)

**Risk Level**: Low-Medium (requires public release decision)

**Prerequisites**:
- Phase 1 completed and stable
- Package version >= 0.1.0 (out of pre-alpha)
- API stability commitment
- License review (currently MIT)
- Security review of dependencies

### 3.1 Pre-Publication Preparation

#### 3.1.1 Version Bump

Update package version to indicate stability:

**File**: `ai_energy_benchmarks/VERSION.txt`
```
0.1.0
```

**File**: `ai_energy_benchmarks/pyproject.toml`
```toml
[project]
name = "ai_energy_benchmarks"
version = "0.1.0"  # Updated from 0.0.1
description = "Modular benchmarking framework for AI energy measurements"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}

classifiers = [
    "Development Status :: 3 - Alpha",  # Updated from Pre-Alpha
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
```

#### 3.1.2 Enhance README for PyPI

**File**: `ai_energy_benchmarks/README.md`

Add at the top:

```markdown
# AI Energy Benchmarks

[![PyPI version](https://badge.fury.io/py/ai-energy-benchmarks.svg)](https://badge.fury.io/py/ai-energy-benchmarks)
[![Python Versions](https://img.shields.io/pypi/pyversions/ai-energy-benchmarks.svg)](https://pypi.org/project/ai-energy-benchmarks/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modular benchmarking framework for measuring AI model energy consumption and carbon emissions across different inference backends.
```

Update installation section:

```markdown
### Installation

#### From PyPI (Recommended)

```bash
# Basic installation
pip install ai-energy-benchmarks

# With PyTorch support
pip install ai-energy-benchmarks[pytorch]

# With all optional dependencies
pip install ai-energy-benchmarks[all]
```

#### From Source

```bash
git clone https://github.com/neuralwatt/ai_energy_benchmarks.git
cd ai_energy_benchmarks
pip install -e .
```
```

#### 3.1.3 Add CHANGELOG

**File**: `ai_energy_benchmarks/CHANGELOG.md`

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-10-12

### Added
- Initial public release
- vLLM backend for high-performance inference
- PyTorch backend for direct model inference
- CodeCarbon integration for energy and emissions tracking
- YAML-based configuration system
- HuggingFace datasets integration
- CSV reporter for benchmark results
- Docker support for containerized benchmarks
- CLI tool: `ai-energy-benchmark`
- Comprehensive test suite with pytest
- Code quality tools: ruff, black, mypy
- Documentation and examples

### Changed
- Package structure optimized for PyPI distribution
- Improved error handling and logging
- Enhanced configuration validation

### Fixed
- Various bug fixes from internal testing

## [0.0.1] - 2025-09-15

### Added
- Initial POC implementation
- Basic PyTorch and vLLM backends
- CodeCarbon metrics collection
- Internal testing and validation
```

#### 3.1.4 Add CONTRIBUTING Guide

**File**: `ai_energy_benchmarks/CONTRIBUTING.md`

```markdown
# Contributing to AI Energy Benchmarks

We welcome contributions! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ai_energy_benchmarks.git
   cd ai_energy_benchmarks
   ```

3. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -e ".[all]"
   pre-commit install
   ```

## Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following our code standards:
   - Use Black for formatting: `black ai_energy_benchmarks/`
   - Run linting: `ruff check ai_energy_benchmarks/`
   - Type checking: `mypy ai_energy_benchmarks/`

3. Add tests for new functionality:
   ```bash
   pytest tests/
   ```

4. Run the full test suite:
   ```bash
   pytest --cov=ai_energy_benchmarks --cov-report=html
   ```

5. Commit your changes:
   ```bash
   git commit -m "feat: add new feature"
   ```
   Follow [Conventional Commits](https://www.conventionalcommits.org/) format.

6. Push and create a pull request

## Code Standards

- **Python Version**: 3.10+
- **Formatting**: Black (line length 100)
- **Linting**: Ruff
- **Type Hints**: Required for public APIs
- **Documentation**: Docstrings for all public functions/classes
- **Tests**: Required for all new features

## Pull Request Process

1. Update documentation if needed
2. Add entry to CHANGELOG.md
3. Ensure all tests pass
4. Request review from maintainers
5. Address review feedback
6. Wait for approval and merge

## Reporting Issues

Use GitHub Issues to report bugs or request features. Include:
- Clear description of the issue
- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Environment details (OS, Python version, GPU model)
- Relevant logs or error messages

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
```

### 3.2 PyPI Account Setup

#### 3.2.1 Create PyPI Account

1. Register at https://pypi.org/account/register/
2. Enable Two-Factor Authentication (required)
3. Create API token:
   - Go to Account Settings → API tokens
   - Create token with scope: "Entire account"
   - Save token securely (show once only)

#### 3.2.2 Configure PyPI Credentials

**File**: `~/.pypirc`

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR-API-TOKEN-HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR-TESTPYPI-TOKEN-HERE
```

**Security**: Set proper permissions:
```bash
chmod 600 ~/.pypirc
```

### 3.3 Test Publication to TestPyPI

Before publishing to production PyPI, test on TestPyPI:

#### 3.3.1 Build Package

```bash
cd ai_energy_benchmarks

# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build wheel and source distribution
python -m build

# Verify build
ls -lh dist/
# Should see:
# ai_energy_benchmarks-0.1.0-py3-none-any.whl
# ai_energy_benchmarks-0.1.0.tar.gz
```

#### 3.3.2 Check Package Quality

```bash
# Install twine if not present
pip install twine

# Check package for common issues
twine check dist/*

# Should output:
# Checking dist/ai_energy_benchmarks-0.1.0-py3-none-any.whl: PASSED
# Checking dist/ai_energy_benchmarks-0.1.0.tar.gz: PASSED
```

#### 3.3.3 Upload to TestPyPI

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Output will show:
# Uploading distributions to https://test.pypi.org/legacy/
# Uploading ai_energy_benchmarks-0.1.0-py3-none-any.whl
# Uploading ai_energy_benchmarks-0.1.0.tar.gz
#
# View at:
# https://test.pypi.org/project/ai-energy-benchmarks/0.1.0/
```

#### 3.3.4 Test Installation from TestPyPI

```bash
# Create fresh test environment
python -m venv test_pypi_env
source test_pypi_env/bin/activate

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    ai-energy-benchmarks

# Test basic functionality
python -c "from ai_energy_benchmarks.runner import BenchmarkRunner; print('✓ Import successful')"
ai-energy-benchmark --help

# Test with extras
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    ai-energy-benchmarks[pytorch]

# Run a simple benchmark
python -c "
from ai_energy_benchmarks.runner import run_benchmark_from_config
print('✓ Package installation successful')
"

# Cleanup
deactivate
rm -rf test_pypi_env
```

### 3.4 Production Publication to PyPI

Once TestPyPI validation passes:

#### 3.4.1 Final Pre-Publication Checklist

- [ ] All tests passing (`pytest`)
- [ ] Documentation complete and accurate
- [ ] CHANGELOG.md updated
- [ ] Version bumped appropriately
- [ ] LICENSE file present (MIT)
- [ ] README.md formatted correctly
- [ ] TestPyPI installation validated
- [ ] Security review completed
- [ ] Team approval obtained

#### 3.4.2 Publish to PyPI

```bash
cd ai_energy_benchmarks

# Upload to production PyPI
twine upload dist/*

# Output will show:
# Uploading distributions to https://upload.pypi.org/legacy/
# Uploading ai_energy_benchmarks-0.1.0-py3-none-any.whl
# Uploading ai_energy_benchmarks-0.1.0.tar.gz
#
# View at:
# https://pypi.org/project/ai-energy-benchmarks/0.1.0/
```

#### 3.4.3 Verify Publication

```bash
# Create fresh environment
python -m venv verify_env
source verify_env/bin/activate

# Install from PyPI
pip install ai-energy-benchmarks

# Verify installation
pip show ai-energy-benchmarks
python -c "import ai_energy_benchmarks; print(ai_energy_benchmarks.__version__)"
ai-energy-benchmark --help

# Test with extras
pip install ai-energy-benchmarks[pytorch]

deactivate
rm -rf verify_env
```

#### 3.4.4 Create GitHub Release

1. Tag the release:
   ```bash
   git tag -a v0.1.0 -m "Release version 0.1.0"
   git push origin v0.1.0
   ```

2. Create GitHub Release:
   - Go to repository → Releases → Create new release
   - Select tag: v0.1.0
   - Title: "AI Energy Benchmarks v0.1.0"
   - Description: Copy from CHANGELOG.md
   - Attach wheel and source distribution from `dist/`

### 3.5 Update Dependent Projects

#### 3.5.1 Update AIEnergyScore

**File**: `AIEnergyScore/requirements.txt`

Add:
```
ai-energy-benchmarks[pytorch]>=0.1.0,<0.2.0
```

**File**: `AIEnergyScore/Dockerfile`

Update:
```dockerfile
# Install ai_energy_benchmarks from PyPI
RUN pip install 'ai-energy-benchmarks[pytorch]>=0.1.0,<0.2.0'
```

**File**: `AIEnergyScore/build.sh`

Simplify (no longer need to build wheel):
```bash
#!/bin/bash
# Build AIEnergyScore Docker image

set -e

echo "Building AIEnergyScore Docker image..."

docker build \
    -f Dockerfile \
    -t ai_energy_score \
    .

echo ""
echo "✓ Docker image 'ai_energy_score' built successfully"
```

#### 3.5.2 Update neuralwatt_cloud (Optional)

If programmatic usage is desired:

**File**: `neuralwatt_cloud/requirements.txt`

Add:
```
ai-energy-benchmarks>=0.1.0,<0.2.0
```

Then can use as library:
```python
from ai_energy_benchmarks import create_benchmark

# Use in Python scripts
benchmark = create_benchmark(config_path="benchmark_config.yaml")
results = benchmark.run()
```

### 3.6 Documentation Updates

#### 3.6.1 Update Main README

**File**: `ai_energy_benchmarks/README.md`

Add badges at top (already shown in 3.1.2)

Add "Installation from PyPI" as primary method

#### 3.6.2 Create PyPI Project Description

PyPI will automatically use `README.md` as the project description. Ensure it:
- Has clear title and badges
- Explains purpose in first paragraph
- Shows installation with `pip install`
- Includes quick start example
- Links to documentation

#### 3.6.3 Update Project URLs

Ensure these are correct in `pyproject.toml`:

```toml
[project.urls]
Homepage = "https://github.com/neuralwatt/ai_energy_benchmarks"
Documentation = "https://github.com/neuralwatt/ai_energy_benchmarks/tree/main/docs"
Repository = "https://github.com/neuralwatt/ai_energy_benchmarks"
Issues = "https://github.com/neuralwatt/ai_energy_benchmarks/issues"
Changelog = "https://github.com/neuralwatt/ai_energy_benchmarks/blob/main/CHANGELOG.md"
```

### 3.7 Ongoing Maintenance

#### 3.7.1 Version Management

Use semantic versioning:
- **MAJOR** (1.0.0): Breaking API changes
- **MINOR** (0.1.0): New features, backward compatible
- **PATCH** (0.1.1): Bug fixes, backward compatible

#### 3.7.2 Release Process

For future releases:

1. Update `VERSION.txt` and `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create feature branch and PR
4. Merge to main after review
5. Build: `python -m build`
6. Test on TestPyPI
7. Publish to PyPI: `twine upload dist/*`
8. Tag release: `git tag v0.x.y`
9. Create GitHub Release

#### 3.7.3 Monitoring

After publication:
- Monitor PyPI download statistics
- Watch GitHub Issues for bug reports
- Track dependent projects (AIEnergyScore, neuralwatt_cloud)
- Respond to community feedback

### 3.8 Benefits of Phase 3

- ✅ Public distribution: `pip install ai-energy-benchmarks`
- ✅ Simplified integration for all projects
- ✅ Version management via PyPI
- ✅ Automatic dependency resolution
- ✅ Community contributions enabled
- ✅ Professional project visibility
- ✅ Standard Python packaging workflow
- ✅ Easier partner/customer adoption

### 3.9 Rollback Plan

If Phase 3 needs to be rolled back:

1. **Yank release from PyPI**:
   ```bash
   # Yanking removes from default search but allows pinned installs
   pip install twine
   twine upload --skip-existing --yanked "Yanked due to [reason]" dist/*
   ```

2. **Revert dependent projects** to Phase 1 wheel method or Phase 0 COPY method

3. **Document issues** in GitHub Issues and `ai_helpers/pip_package_migration_issues.md`

4. **Address concerns** before attempting re-publication

**Note**: Once published to PyPI, a version number cannot be reused. If v0.1.0 is yanked, next release must be v0.1.1 or higher.

---

## Success Criteria

### Phase 1
- [ ] Wheel builds successfully
- [ ] AIEnergyScore Docker build works with wheel
- [ ] All AIEnergyScore backends functional (optimum, pytorch, vllm)
- [ ] Build time improved vs COPY method
- [ ] Documentation updated

### Phase 3
- [ ] Package published to PyPI
- [ ] Installation works: `pip install ai-energy-benchmarks`
- [ ] All optional dependencies installable: `[pytorch]`, `[all]`
- [ ] CLI tool accessible: `ai-energy-benchmark --help`
- [ ] AIEnergyScore updated to use PyPI version
- [ ] Documentation complete
- [ ] GitHub Release created
- [ ] Community can install and use package

---

## Timeline

### Phase 1: Internal Wheel Distribution
- **Day 1**: Create build script, update Dockerfile
- **Day 2**: Testing and validation
- **Total**: 2 days

### Phase 3: Public PyPI Distribution
- **Day 1-2**: Pre-publication preparation (version bump, docs, changelog)
- **Day 3**: PyPI account setup, TestPyPI publication
- **Day 4**: Validation and testing
- **Day 5**: Production PyPI publication
- **Day 6-7**: Update dependent projects, documentation
- **Total**: 7 days

**Overall Timeline**: ~2 weeks from start to full public release

---

## References

- [Python Packaging User Guide](https://packaging.python.org/)
- [PyPI Help](https://pypi.org/help/)
- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
- [Conventional Commits](https://www.conventionalcommits.org/)

---

## Support & Questions

For questions about this migration:
- **GitHub Issues**: https://github.com/neuralwatt/ai_energy_benchmarks/issues
- **Internal Documentation**: `ai_helpers/backend_switching_strategy.md`
- **Email**: info@neuralwatt.com

---

*Document Version: 1.0*
*Last Updated: 2025-10-12*
*Author: NeuralWatt Team*
