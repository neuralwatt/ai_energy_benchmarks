# Gap Analysis & Implementation Plan: ai_energy_benchmarks pip Deployment

**Date:** 2025-10-20
**Status:** Planning
**Version:** 0.0.1

## Executive Summary

This document analyzes the current state of ai_energy_benchmarks for pip-based deployment and identifies gaps that need to be addressed. The goal is to enable AIEnergyScore Docker to install ai_energy_benchmarks from pip rather than local source, with a dual-track release strategy:

- **PPE Branch** → TestPyPI for pre-production testing
- **Main Branch** → Production PyPI for official releases

## Current State ✅

### What's Working Well
- ✅ Package structure ready (pyproject.toml, setup.py, VERSION.txt)
- ✅ Wheel builds successfully (0.0.1 available in dist/)
- ✅ Pre-commit hooks configured (ruff, black, mypy)
- ✅ MIT License in place
- ✅ Comprehensive README.md with usage documentation
- ✅ GitHub Actions for Docker publishing exists (.github/workflows/ghcr-publish.yml)
- ✅ Optional dependencies structure (pytorch, dev, all)
- ✅ Basic package metadata in pyproject.toml
- ✅ Build script (build_wheel.sh) working

### Current Installation Method
```bash
# Local source installation (current)
cd /ai_energy_benchmarks && pip install .

# Target: pip-based installation (goal)
pip install ai_energy_benchmarks
# or from wheel
pip install ai_energy_benchmarks-0.0.1-py3-none-any.whl
```

## Gaps Identified ❌

### Critical Gaps (Must-Have for pip deployment)

#### 1. Missing `py.typed` Marker File
**Impact:** High
**Status:** ❌ Missing

Python packages that want to support type checking need a `py.typed` marker file.

**Location:** `ai_energy_benchmarks/py.typed`
**Content:** Empty file or "partial\n"
**Reference:** PEP 561

#### 2. Package Metadata Incomplete
**Impact:** Medium
**Status:** ⚠️ Partial

Current pyproject.toml doesn't include long_description from README.

**Fix:**
```toml
[project]
dynamic = ["version"]
readme = "README.md"  # ✅ Already present
# Ensures README is used as long_description on PyPI
```

#### 3. Config Files Not Verified in Package
**Impact:** High
**Status:** ⚠️ Needs Verification

The `reasoning_formats.yaml` file needs to be included in the wheel.

**Current declaration:**
```toml
[tool.setuptools.package-data]
ai_energy_benchmarks = ["py.typed", "config/*.yaml"]
```

**Action Required:** Verify the YAML file is actually included in built wheel.

#### 4. AIEnergyScore Dockerfile Uses Local Source
**Impact:** High
**Status:** ❌ Blocking

Current Dockerfile (line 26-27):
```dockerfile
COPY ai_energy_benchmarks /ai_energy_benchmarks
RUN cd /ai_energy_benchmarks && pip install .
```

**Target:**
```dockerfile
# Option A: Install from pre-built wheel (recommended for now)
COPY ai_energy_benchmarks/dist/*.whl /tmp/
RUN pip install /tmp/ai_energy_benchmarks-*.whl

# Option B: Install from TestPyPI (PPE testing)
RUN pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    ai_energy_benchmarks==0.0.1rc1

# Option C: Install from PyPI (production - future)
RUN pip install ai_energy_benchmarks==0.0.1

# Option D: Install from PyPI with extras (production)
RUN pip install ai_energy_benchmarks[pytorch]==0.0.1
```

#### 5. Missing `__init__.py` in Package Root
**Impact:** High
**Status:** ❌ Critical

**Error Found:** No `__init__.py` in `ai_energy_benchmarks/` directory root.

**Location:** `ai_energy_benchmarks/__init__.py`
**Required Content:**
```python
"""AI Energy Benchmarks - Modular benchmarking framework for AI energy measurements."""

__version__ = "0.0.1"
__author__ = "NeuralWatt"
__license__ = "MIT"

# Import key classes for convenience
from ai_energy_benchmarks.runner import BenchmarkRunner
from ai_energy_benchmarks.config.parser import BenchmarkConfig

__all__ = ["BenchmarkRunner", "BenchmarkConfig", "__version__"]
```

### Important Gaps (Best Practices)

#### 6. No CHANGELOG.md
**Impact:** Medium
**Status:** ❌ Missing

Version history tracking is essential for package management.

**Recommended Format:** Keep a Changelog (keepachangelog.com)

**Template:**
```markdown
# Changelog

## [Unreleased]

## [0.0.1] - 2025-10-20
### Added
- Initial release
- vLLM backend support
- PyTorch backend support
- Reasoning format system
- Multi-GPU support
```

#### 7. No PyPI Publishing Workflow
**Impact:** High
**Status:** ❌ Missing

Need separate GitHub Actions workflows for:
- PPE branch → TestPyPI (automated testing)
- Main branch → PyPI (production releases)

#### 8. No CI/CD Testing Workflow
**Impact:** Medium
**Status:** ❌ Missing

Automated testing on push/PR (pytest, ruff, mypy, black).

#### 9. No MANIFEST.in
**Impact:** Low
**Status:** ⚠️ Optional

While not strictly required with pyproject.toml, MANIFEST.in provides explicit control.

**Recommended Content:**
```
include README.md
include LICENSE
include VERSION.txt
include pyproject.toml
recursive-include ai_energy_benchmarks/config *.yaml
recursive-include ai_energy_benchmarks *.py
recursive-include ai_energy_benchmarks py.typed
```

#### 10. No Installation Verification Tests
**Impact:** Medium
**Status:** ❌ Missing

Automated tests to verify package installs correctly.

## PPE/Main Release Strategy

### Overview

The ai_energy_benchmarks project will use a **dual-track release strategy** to enable safe testing and production deployments:

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│  PPE Branch │  ────>  │   TestPyPI   │  ────>  │ PPE Testing │
└─────────────┘         └──────────────┘         └─────────────┘
                              │
                              │ (validate)
                              ↓
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│ Main Branch │  ────>  │  PyPI (Prod) │  ────>  │ Production  │
└─────────────┘         └──────────────┘         └─────────────┘
```

### PPE Branch Strategy

**Purpose:** Pre-production testing and validation

**Workflow:**
1. Developers push code to `ppe` branch
2. GitHub Actions automatically:
   - Runs all tests (pytest, ruff, mypy)
   - Bumps version to development/release candidate format
   - Builds wheel
   - Publishes to TestPyPI
3. AIEnergyScore Docker can pull from TestPyPI for integration testing
4. Team validates functionality in PPE environment

**Version Format:**
- Development builds: `0.0.1.devN` (where N is build number)
- Release candidates: `0.0.1rc1`, `0.0.1rc2`, etc.
- Example: `0.0.2.dev42` or `0.1.0rc1`

**Installation from TestPyPI:**
```bash
# Install from TestPyPI with fallback to PyPI for dependencies
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    ai_energy_benchmarks==0.0.1rc1

# With extras
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    "ai_energy_benchmarks[pytorch]==0.0.1rc1"
```

**Docker Usage (PPE):**
```dockerfile
# In AIEnergyScore/Dockerfile for PPE testing
ARG AI_ENERGY_BENCHMARKS_VERSION=0.0.1rc1
RUN pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    ai_energy_benchmarks==${AI_ENERGY_BENCHMARKS_VERSION}
```

**Automation:**
```yaml
# .github/workflows/test-and-publish-ppe.yml
name: PPE - Test and Publish to TestPyPI

on:
  push:
    branches:
      - ppe

jobs:
  test-and-publish:
    runs-on: ubuntu-latest
    steps:
      - name: Run tests
        # pytest, ruff, mypy, etc.

      - name: Bump version to dev/rc
        # Auto-increment dev or rc version

      - name: Build package
        run: python -m build

      - name: Publish to TestPyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.TEST_PYPI_TOKEN }}
        run: twine upload --repository testpypi dist/*
```

### Main Branch Strategy

**Purpose:** Production releases

**Workflow:**
1. Code is merged from `ppe` to `main` after validation
2. Release manager creates version tag (e.g., `v0.0.1`)
3. GitHub Actions automatically:
   - Verifies all tests pass
   - Builds wheel with production version
   - Publishes to production PyPI
   - Creates GitHub release with CHANGELOG
4. AIEnergyScore production Docker pulls from PyPI

**Version Format:**
- Semantic versioning: `MAJOR.MINOR.PATCH`
- Examples: `0.0.1`, `0.1.0`, `1.0.0`
- No dev/rc suffixes in production

**Release Checklist:**
- [ ] All tests pass on ppe branch
- [ ] Integration testing with AIEnergyScore complete
- [ ] CHANGELOG.md updated
- [ ] Version number bumped in VERSION.txt
- [ ] Code reviewed and approved
- [ ] Merged ppe → main
- [ ] Tag created: `git tag v0.0.1`
- [ ] Tag pushed: `git push origin v0.0.1`

**Installation from PyPI:**
```bash
# Production installation
pip install ai_energy_benchmarks==0.0.1

# With extras
pip install "ai_energy_benchmarks[pytorch]==0.0.1"
```

**Docker Usage (Production):**
```dockerfile
# In AIEnergyScore/Dockerfile for production
ARG AI_ENERGY_BENCHMARKS_VERSION=0.0.1
RUN pip install ai_energy_benchmarks==${AI_ENERGY_BENCHMARKS_VERSION}
```

**Automation:**
```yaml
# .github/workflows/publish-main.yml
name: Main - Publish to PyPI

on:
  push:
    tags:
      - 'v*'

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - name: Validate tag format
        # Ensure tag matches v0.0.1 format

      - name: Run all tests
        # pytest, ruff, mypy - MUST PASS

      - name: Build package
        run: python -m build

      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
        run: twine upload dist/*

      - name: Create GitHub Release
        # Auto-create release with CHANGELOG excerpt
```

### Version Bumping Strategy

**Tools:** Use `bumpver` or `bump2version` for automated version management

**Configuration (bumpver.toml):**
```toml
[bumpver]
current_version = "0.0.1"
version_pattern = "MAJOR.MINOR.PATCH[PYTAGNUM]"
commit_message = "Bump version {old_version} → {new_version}"
commit = true
tag = true
push = false

[bumpver.file_patterns]
"VERSION.txt" = [
    "{version}",
]
"ai_energy_benchmarks/__init__.py" = [
    '__version__ = "{version}"',
]
"pyproject.toml" = [
    'version = "{version}"',
]
```

**Version Bump Commands:**
```bash
# PPE branch - development builds
bumpver update --tag dev --tag-num

# PPE branch - release candidates
bumpver update --tag rc --tag-num

# Main branch - patch release
bumpver update --patch

# Main branch - minor release
bumpver update --minor

# Main branch - major release
bumpver update --major
```

### Repository Setup Required

#### TestPyPI Account
1. Create account at https://test.pypi.org/
2. Generate API token
3. Add to GitHub secrets as `TEST_PYPI_TOKEN`

#### PyPI Account
1. Create account at https://pypi.org/
2. Generate API token
3. Add to GitHub secrets as `PYPI_TOKEN`

#### GitHub Configuration
```bash
# Add secrets in GitHub repository settings
Settings → Secrets and variables → Actions

Required secrets:
- TEST_PYPI_TOKEN  # For TestPyPI publishing
- PYPI_TOKEN       # For PyPI publishing
```

### Testing the Release Process

#### Test TestPyPI Publishing
```bash
# From ppe branch
git checkout ppe
git push origin ppe

# Verify workflow runs
# Check https://test.pypi.org/project/ai-energy-benchmarks/

# Test installation
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    ai_energy_benchmarks
```

#### Test PyPI Publishing
```bash
# From main branch (after ppe validation)
git checkout main
git merge ppe
git tag v0.0.1
git push origin v0.0.1

# Verify workflow runs
# Check https://pypi.org/project/ai-energy-benchmarks/

# Test installation
pip install ai_energy_benchmarks==0.0.1
```

## Implementation Plan

### Phase 1: Package Preparation (Critical - Required for pip deployment)

**Estimated Time:** 2-3 hours

#### Task 1.1: Add Type Hints Marker
- [ ] Create `ai_energy_benchmarks/py.typed` (empty file)
- [ ] Verify it's listed in pyproject.toml package-data
- [ ] Test with mypy

#### Task 1.2: Create Package __init__.py
- [ ] Create `ai_energy_benchmarks/__init__.py`
- [ ] Add version, author, license metadata
- [ ] Export key classes (BenchmarkRunner, BenchmarkConfig)
- [ ] Add __all__ for clean imports

#### Task 1.3: Create MANIFEST.in
- [ ] Create `MANIFEST.in` with explicit file inclusions
- [ ] Ensure README, LICENSE, VERSION.txt included
- [ ] Ensure config YAML files included

#### Task 1.4: Verify Package Metadata
- [ ] Confirm README.md is used as long_description
- [ ] Check all URLs in pyproject.toml are correct
- [ ] Verify classifiers are appropriate

#### Task 1.5: Build and Verify Wheel
```bash
cd ai_energy_benchmarks
./build_wheel.sh
unzip -l dist/ai_energy_benchmarks-0.0.1-py3-none-any.whl | grep reasoning_formats.yaml
unzip -l dist/ai_energy_benchmarks-0.0.1-py3-none-any.whl | grep py.typed
```

#### Task 1.6: Test Installation in Clean Environment
```bash
# Create clean venv
python3 -m venv /tmp/test_install
source /tmp/test_install/bin/activate

# Install from wheel
pip install ai_energy_benchmarks/dist/ai_energy_benchmarks-0.0.1-py3-none-any.whl

# Verify imports work
python3 -c "from ai_energy_benchmarks import BenchmarkRunner; print('Success')"
python3 -c "import ai_energy_benchmarks; print(ai_energy_benchmarks.__version__)"

# Verify config files are accessible
python3 -c "from ai_energy_benchmarks.formatters.registry import FormatterRegistry; r = FormatterRegistry(); print('Config loaded')"

# Clean up
deactivate
rm -rf /tmp/test_install
```

### Phase 2: AIEnergyScore Docker Migration (Critical)

**Estimated Time:** 1-2 hours

#### Task 2.1: Update AIEnergyScore Build Process
- [ ] Update `AIEnergyScore/build.sh` to build ai_energy_benchmarks wheel first
- [ ] Copy wheel to AIEnergyScore/dist/ or include in build context

#### Task 2.2: Create Docker Build Strategy Options

**Option A: Local wheel (recommended for immediate deployment)**
```dockerfile
# Build ai_energy_benchmarks wheel first (in build.sh)
# Then in Dockerfile:
COPY ai_energy_benchmarks/dist/ai_energy_benchmarks-*.whl /tmp/
RUN pip install /tmp/ai_energy_benchmarks-*.whl && rm -rf /tmp/*.whl
```

**Option B: TestPyPI (for PPE testing)**
```dockerfile
ARG AI_ENERGY_BENCHMARKS_VERSION=0.0.1rc1
RUN pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    ai_energy_benchmarks==${AI_ENERGY_BENCHMARKS_VERSION}
```

**Option C: Production PyPI (for main branch)**
```dockerfile
ARG AI_ENERGY_BENCHMARKS_VERSION=0.0.1
RUN pip install ai_energy_benchmarks==${AI_ENERGY_BENCHMARKS_VERSION}
```

**Option D: Multi-stage build (most flexible)**
```dockerfile
# Stage 1: Build wheel
FROM python:3.11-slim as builder
WORKDIR /build
COPY ai_energy_benchmarks /build/ai_energy_benchmarks
RUN cd /build/ai_energy_benchmarks && \
    pip install build && \
    python -m build

# Stage 2: Install
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime
COPY --from=builder /build/ai_energy_benchmarks/dist/*.whl /tmp/
RUN pip install /tmp/ai_energy_benchmarks-*.whl && rm -rf /tmp/*.whl
```

#### Task 2.3: Create Branch-Specific Dockerfiles
- [ ] Create `AIEnergyScore/Dockerfile.ppe` for PPE testing (TestPyPI)
- [ ] Create `AIEnergyScore/Dockerfile.main` for production (PyPI)
- [ ] Update `AIEnergyScore/Dockerfile` for local wheel (default)

#### Task 2.4: Update Documentation
- [ ] Update AIEnergyScore/README.md with new build process
- [ ] Update build.sh comments
- [ ] Add troubleshooting section for wheel builds

#### Task 2.5: Test Docker Build
```bash
cd ~/src

# Test local wheel installation
./AIEnergyScore/build.sh
docker run --gpus all ai_energy_score --help

# Test TestPyPI installation (after publishing to TestPyPI)
docker build -f AIEnergyScore/Dockerfile.ppe -t ai_energy_score:ppe .
docker run --gpus all ai_energy_score:ppe --help
```

### Phase 3: Release Automation (Critical for PPE/Main Strategy)

**Estimated Time:** 3-4 hours

#### Task 3.1: Set Up PyPI Accounts
- [ ] Create TestPyPI account (https://test.pypi.org/)
- [ ] Create PyPI account (https://pypi.org/)
- [ ] Generate API tokens for both
- [ ] Add tokens to GitHub secrets

#### Task 3.2: Configure Version Bumping
- [ ] Install bumpver: `pip install bumpver`
- [ ] Create `bumpver.toml` configuration
- [ ] Test version bumping commands
- [ ] Document version bump workflow

#### Task 3.3: Create PPE Publishing Workflow
Create `.github/workflows/test-and-publish-ppe.yml`:
```yaml
name: PPE - Test and Publish to TestPyPI

on:
  push:
    branches:
      - ppe

jobs:
  test-and-publish:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
          pip install build twine bumpver

      - name: Run ruff check
        run: ruff check .

      - name: Run ruff format check
        run: ruff format --check .

      - name: Run mypy
        run: mypy ai_energy_benchmarks/

      - name: Run tests
        run: pytest --cov=ai_energy_benchmarks

      - name: Bump version to dev/rc
        run: |
          # Increment dev version: 0.0.1 → 0.0.1.dev1 → 0.0.1.dev2
          bumpver update --tag dev --tag-num --no-commit --no-tag

      - name: Build package
        run: python -m build

      - name: Publish to TestPyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.TEST_PYPI_TOKEN }}
        run: twine upload --repository testpypi dist/*
```

#### Task 3.4: Create Main Publishing Workflow
Create `.github/workflows/publish-main.yml`:
```yaml
name: Main - Publish to PyPI

on:
  push:
    tags:
      - 'v*'

jobs:
  validate-and-publish:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Validate tag format
        run: |
          if [[ ! "${GITHUB_REF}" =~ ^refs/tags/v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            echo "Invalid tag format. Use v0.0.1 format."
            exit 1
          fi

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
          pip install build twine

      - name: Run ALL quality checks (MUST PASS)
        run: |
          ruff check .
          ruff format --check .
          mypy ai_energy_benchmarks/
          pytest --cov=ai_energy_benchmarks --cov-fail-under=70

      - name: Build package
        run: python -m build

      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
        run: twine upload dist/*

      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          body_path: CHANGELOG.md
          files: dist/*
```

#### Task 3.5: Create Testing Workflow (Runs on all branches)
Create `.github/workflows/test.yml`:
```yaml
name: Tests

on:
  push:
    branches: ["**"]
  pull_request:
    branches: ["**"]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run ruff
        run: ruff check .

      - name: Run ruff format check
        run: ruff format --check .

      - name: Run mypy
        run: mypy ai_energy_benchmarks/

      - name: Run pytest
        run: pytest --cov=ai_energy_benchmarks
```

### Phase 4: Documentation (Important)

**Estimated Time:** 2-3 hours

#### Task 4.1: Create CHANGELOG.md
```markdown
# Changelog

All notable changes to ai_energy_benchmarks will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.1] - 2025-10-20

### Added
- Initial release of ai_energy_benchmarks
- vLLM backend for high-performance inference
- PyTorch backend for direct model inference
- Multi-GPU support with automatic model distribution
- Reasoning format system with unified configuration
- Support for gpt-oss, DeepSeek, SmolLM, Qwen, Hunyuan models
- CodeCarbon integration for energy metrics
- CSV reporter for benchmark results
- Comprehensive configuration system
- Docker support
- Pre-commit hooks for code quality

### Documentation
- Comprehensive README with examples
- API documentation
- Configuration guides
- Multi-GPU deployment guide
```

#### Task 4.2: Create PUBLISHING.md
```markdown
# Publishing Guide

## Overview

ai_energy_benchmarks uses a dual-track release strategy:
- **PPE Branch** → TestPyPI for testing
- **Main Branch** → PyPI for production

## PPE Release Process

[Detailed steps for PPE releases...]

## Production Release Process

[Detailed steps for production releases...]

## Version Numbering

[Semantic versioning guidelines...]
```

#### Task 4.3: Update README.md
Add installation sections:
```markdown
## Installation

### From PyPI (Production)
pip install ai_energy_benchmarks
pip install "ai_energy_benchmarks[pytorch]"

### From TestPyPI (Testing)
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    ai_energy_benchmarks

### From Source
git clone https://github.com/neuralwatt/ai_energy_benchmarks
cd ai_energy_benchmarks
pip install -e ".[dev]"
```

#### Task 4.4: Add Installation Test Script
Create `ai_helpers/test_installation.sh`:
```bash
#!/bin/bash
# Test that ai_energy_benchmarks installs and works correctly
set -e

echo "Testing ai_energy_benchmarks installation..."

TMPDIR=$(mktemp -d)
python3 -m venv "$TMPDIR/venv"
source "$TMPDIR/venv/bin/activate"

pip install dist/ai_energy_benchmarks-*.whl

python3 -c "from ai_energy_benchmarks import BenchmarkRunner"
python3 -c "import ai_energy_benchmarks; assert ai_energy_benchmarks.__version__ == '0.0.1'"
python3 -c "from ai_energy_benchmarks.formatters.registry import FormatterRegistry; FormatterRegistry()"

echo "✓ Installation test passed"

deactivate
rm -rf "$TMPDIR"
```

## Testing Checklist

### Pre-Deployment Testing (Phase 1)
- [ ] Wheel builds without errors
- [ ] All package files included in wheel (verify with unzip -l)
- [ ] py.typed marker present
- [ ] Config YAML files present
- [ ] __init__.py exports work correctly
- [ ] Installation in clean venv succeeds
- [ ] Imports work after installation
- [ ] Config files load correctly after installation
- [ ] Pre-commit hooks pass (ruff, black, mypy)
- [ ] All unit tests pass

### Docker Testing (Phase 2)
- [ ] AIEnergyScore builds with wheel-based installation
- [ ] Docker image size acceptable
- [ ] ai_energy_benchmarks imports work in container
- [ ] Benchmarks run successfully in container
- [ ] Both PyTorch and vLLM backends work

### PPE Testing (Phase 3)
- [ ] GitHub Actions workflow runs on ppe push
- [ ] Package publishes to TestPyPI automatically
- [ ] Version format is correct (.devN or rcN)
- [ ] Can install from TestPyPI
- [ ] AIEnergyScore can use TestPyPI version
- [ ] All functionality works from TestPyPI install

### Production Release Testing (Phase 3)
- [ ] GitHub Actions workflow runs on tag push
- [ ] All quality checks pass
- [ ] Package publishes to PyPI
- [ ] Version format is correct (semantic versioning)
- [ ] Can install from PyPI
- [ ] AIEnergyScore can use PyPI version
- [ ] GitHub release created with CHANGELOG

## Risk Assessment

### High Risk Items
1. **Missing config files in wheel** - Could break formatter system
2. **Import errors** - Missing __init__.py breaks package structure
3. **Docker build failures** - Incorrect wheel path or permissions
4. **TestPyPI dependency resolution** - Some deps may not be on TestPyPI
5. **Version conflicts** - PPE and production versions could conflict

### Mitigation Strategies
1. Always verify wheel contents with `unzip -l` before deployment
2. Test installation in clean environment before publishing
3. Keep local source installation as fallback in Docker
4. Use `--extra-index-url https://pypi.org/simple/` with TestPyPI
5. Use clear version naming (dev/rc for PPE, semantic for production)
6. Maintain separate Dockerfiles for PPE and production
7. Test PPE releases before merging to main

## Success Criteria

### Minimum Viable Product (MVP) - Phase 1 & 2
- ✅ Wheel builds and includes all necessary files
- ✅ Package installs successfully from wheel
- ✅ AIEnergyScore Docker uses pip/wheel installation
- ✅ All functionality works after pip installation

### PPE Deployment Ready - Phase 3 (PPE)
- ✅ Automated publishing to TestPyPI on ppe push
- ✅ Can install from TestPyPI
- ✅ Integration testing with AIEnergyScore successful
- ✅ Version bumping automated

### Production Ready - Phase 3 (Main)
- ✅ Published to PyPI
- ✅ CI/CD workflows operational
- ✅ Documentation complete and accurate
- ✅ CHANGELOG maintained
- ✅ Version management automated
- ✅ GitHub releases created automatically

## Timeline Estimate

- **Phase 1 (Critical):** 2-3 hours - Package foundation
- **Phase 2 (Critical):** 1-2 hours - Docker migration
- **Phase 3 (Critical for PPE/Main):** 3-4 hours - Release automation
- **Phase 4 (Important):** 2-3 hours - Documentation

**Total Time:** 8-12 hours for complete implementation

**Minimum Time to PPE Deployment:** 6-9 hours (Phases 1, 2, 3)
**Minimum Time to Production:** 8-12 hours (All phases)

## Next Steps

### Immediate Actions
1. Create missing package files (py.typed, __init__.py, MANIFEST.in)
2. Verify wheel includes all necessary files
3. Test installation in clean environment

### Short-Term Actions
4. Update AIEnergyScore Dockerfile to use wheel
5. Set up TestPyPI and PyPI accounts
6. Configure GitHub secrets

### Medium-Term Actions
7. Create GitHub Actions workflows for PPE and Main
8. Set up version bumping automation
9. Create CHANGELOG.md and PUBLISHING.md

### Long-Term Actions
10. Establish release cadence
11. Monitor PyPI download stats
12. Collect feedback from users

## Release Workflow Summary

```
Developer Flow:
1. Work on feature branch
2. Merge to ppe branch
3. GitHub Actions auto-publishes to TestPyPI
4. Test with AIEnergyScore in PPE environment
5. If validated, merge ppe → main
6. Create version tag (v0.0.1)
7. GitHub Actions auto-publishes to PyPI
8. Production systems use PyPI version

Branch Strategy:
feature → ppe (TestPyPI) → main (PyPI) → production
```

## References

- [PEP 517 - Build System](https://peps.python.org/pep-0517/)
- [PEP 518 - Build System Requirements](https://peps.python.org/pep-0518/)
- [PEP 561 - Type Hints](https://peps.python.org/pep-0561/)
- [PyPA Packaging Guide](https://packaging.python.org/)
- [TestPyPI](https://test.pypi.org/)
- [Keep a Changelog](https://keepachangelog.com/)
- [Semantic Versioning](https://semver.org/)
- [GitHub Actions Publishing](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python#publishing-to-package-registries)

---

**Document Version:** 2.0
**Last Updated:** 2025-10-20
**Author:** Claude Code Analysis
**Status:** Ready for Implementation
