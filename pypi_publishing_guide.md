# PyPI Publishing Guide for ai_energy_benchmarks

**Quick Start Guide for Publishing to PyPI/TestPyPI**

## Prerequisites

### 1. Install Publishing Tools
```bash
pip install twine build
```

### 2. Create PyPI Accounts

#### TestPyPI (for testing - recommended first!)
1. Go to https://test.pypi.org/account/register/
2. Create an account
3. Verify your email address
4. Go to https://test.pypi.org/manage/account/token/
5. Click "Add API token"
   - Token name: `test_ai_energy_benchmarks_upload`
   - Scope: "Entire account" (or specific to project after first upload)
6. **SAVE THE TOKEN** - you'll only see it once!
   - Format: `pypi-AgEIcHlwaS...` (starts with `pypi-`)

#### PyPI (for production)
1. Go to https://pypi.org/account/register/
2. Create an account
3. Verify your email address
4. Go to https://pypi.org/manage/account/token/
5. Click "Add API token"
   - Token name: `ai_energy_benchmarks_upload`
   - Scope: "Entire account"
6. **SAVE THE TOKEN**

### 3. Configure Credentials

Create `~/.pypirc`:
```bash
cat > ~/.pypirc << 'EOF'
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR-PRODUCTION-TOKEN-HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR-TEST-TOKEN-HERE
EOF

chmod 600 ~/.pypirc  # Secure the file
```

**Replace the tokens** with your actual tokens from PyPI and TestPyPI!

## Publishing Workflow

### Step 1: Prepare for Release

```bash
cd ~/src/ai_energy_benchmarks

# Ensure you're on the right branch
git checkout ppe    # For TestPyPI testing
# OR
git checkout main   # For production PyPI

# Make sure everything is committed
git status

# Run tests
pytest

# Run linting
.venv/bin/ruff check .
.venv/bin/mypy ai_energy_benchmarks/
```

### Step 2: Update Version (if needed)

Edit `VERSION.txt`:
```bash
# For development/testing (TestPyPI)
echo "0.0.1rc1" > VERSION.txt
# OR
echo "0.0.1.dev1" > VERSION.txt

# For production (PyPI)
echo "0.0.1" > VERSION.txt
```

Also update in `ai_energy_benchmarks/__init__.py`:
```python
__version__ = "0.0.1rc1"  # or "0.0.1"
```

### Step 3: Build the Distribution

```bash
cd ~/src/ai_energy_benchmarks

# Clean old builds
rm -rf dist/ build/ *.egg-info

# Build both wheel and source distribution
python -m build

# Verify what was built
ls -lh dist/
# Should show:
# - ai_energy_benchmarks-0.0.1rc1-py3-none-any.whl
# - ai_energy_benchmarks-0.0.1rc1.tar.gz
```

### Step 4: Upload to TestPyPI (Test First!)

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# You should see:
# Uploading distributions to https://test.pypi.org/legacy/
# Uploading ai_energy_benchmarks-0.0.1rc1-py3-none-any.whl
# Uploading ai_energy_benchmarks-0.0.1rc1.tar.gz
```

### Step 5: Test Installation from TestPyPI

```bash
# Create test environment
python3 -m venv /tmp/test_pypi_install
source /tmp/test_pypi_install/bin/activate

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    ai_energy_benchmarks==0.0.1

# Test it works
python -c "from ai_energy_benchmarks import BenchmarkRunner; print('✓ Success')"
python -c "import ai_energy_benchmarks; print(f'Version: {ai_energy_benchmarks.__version__}')"

# Cleanup
deactivate
rm -rf /tmp/test_pypi_install
```

**Note:** The `--extra-index-url https://pypi.org/simple/` is needed because TestPyPI doesn't have all dependencies. It will get dependencies like `requests`, `datasets`, etc. from the main PyPI.

### Step 6: Upload to Production PyPI

**Only after TestPyPI testing succeeds!**

```bash
cd ~/src/ai_energy_benchmarks

# Switch to main branch
git checkout main

# Update version to production version
echo "0.0.1" > VERSION.txt
# Update __version__ in ai_energy_benchmarks/__init__.py

# Commit version bump
git add VERSION.txt ai_energy_benchmarks/__init__.py
git commit -m "Bump version to 0.0.1"
git tag v0.0.1
git push origin main
git push origin v0.0.1

# Clean and rebuild
rm -rf dist/ build/ *.egg-info
python -m build

# Upload to production PyPI
twine upload dist/*

# You should see:
# Uploading distributions to https://upload.pypi.org/legacy/
# Uploading ai_energy_benchmarks-0.0.1-py3-none-any.whl
# Uploading ai_energy_benchmarks-0.0.1.tar.gz
```

### Step 7: Test Installation from PyPI

```bash
# Create test environment
python3 -m venv /tmp/test_production_install
source /tmp/test_production_install/bin/activate

# Install from production PyPI
pip install ai_energy_benchmarks==0.0.1

# Test it works
python -c "from ai_energy_benchmarks import BenchmarkRunner; print('✓ Success')"
python -c "import ai_energy_benchmarks; print(f'Version: {ai_energy_benchmarks.__version__}')"

# Cleanup
deactivate
rm -rf /tmp/test_production_install
```

## Using Published Package in AIEnergyScore

### From TestPyPI (for ppe branch testing)

Update `AIEnergyScore/Dockerfile`:
```dockerfile
# Install ai_energy_benchmarks from TestPyPI
ARG AI_ENERGY_BENCHMARKS_VERSION=0.0.1rc1
RUN pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    ai_energy_benchmarks==${AI_ENERGY_BENCHMARKS_VERSION}
```

Build:
```bash
cd ~/src/AIEnergyScore
docker build -f Dockerfile -t ai_energy_score:ppe \
    --build-arg AI_ENERGY_BENCHMARKS_VERSION=0.0.1rc1 \
    .
```

### From Production PyPI (for main branch)

Update `AIEnergyScore/Dockerfile`:
```dockerfile
# Install ai_energy_benchmarks from PyPI
ARG AI_ENERGY_BENCHMARKS_VERSION=0.0.1
RUN pip install ai_energy_benchmarks==${AI_ENERGY_BENCHMARKS_VERSION}
```

Build:
```bash
cd ~/src/AIEnergyScore
docker build -f Dockerfile -t ai_energy_score:latest \
    --build-arg AI_ENERGY_BENCHMARKS_VERSION=0.0.1 \
    .
```

## Version Numbering Strategy

### TestPyPI (ppe branch)
- Development: `0.0.1.dev1`, `0.0.1.dev2`, etc.
- Release candidates: `0.0.1rc1`, `0.0.1rc2`, etc.
- Pre-releases: `0.0.1a1`, `0.0.1b1`, etc.

### Production PyPI (main branch)
- Semantic versioning: `MAJOR.MINOR.PATCH`
- Examples: `0.0.1`, `0.1.0`, `1.0.0`
- **No dev/rc/alpha/beta suffixes**

### Bumping Versions

```bash
# Patch release (bug fixes)
0.0.1 → 0.0.2

# Minor release (new features, backwards compatible)
0.0.2 → 0.1.0

# Major release (breaking changes)
0.1.0 → 1.0.0
```

## Complete Example: First Release

### TestPyPI (Testing)
```bash
# 1. Prepare
cd ~/src/ai_energy_benchmarks
git checkout ppe
git pull origin ppe

# 2. Set version
echo "0.0.1rc1" > VERSION.txt
# Edit ai_energy_benchmarks/__init__.py: __version__ = "0.0.1rc1"

# 3. Build
rm -rf dist/ build/ *.egg-info
python -m build

# 4. Upload to TestPyPI
twine upload --repository testpypi dist/*

# 5. Test installation
python3 -m venv /tmp/test_env
source /tmp/test_env/bin/activate
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    ai_energy_benchmarks==0.0.1rc1
python -c "from ai_energy_benchmarks import BenchmarkRunner; print('✓')"
deactivate
rm -rf /tmp/test_env
```

### Production PyPI (After Testing)
```bash
# 1. Prepare
cd ~/src/ai_energy_benchmarks
git checkout main
git pull origin main

# 2. Set version
echo "0.0.1" > VERSION.txt
# Edit ai_energy_benchmarks/__init__.py: __version__ = "0.0.1"

# 3. Commit and tag
git add VERSION.txt ai_energy_benchmarks/__init__.py
git commit -m "Release version 0.0.1"
git tag v0.0.1
git push origin main
git push origin v0.0.1

# 4. Build
rm -rf dist/ build/ *.egg-info
python -m build

# 5. Upload to PyPI
twine upload dist/*

# 6. Test installation
python3 -m venv /tmp/test_env
source /tmp/test_env/bin/activate
pip install ai_energy_benchmarks==0.0.1
python -c "from ai_energy_benchmarks import BenchmarkRunner; print('✓')"
deactivate
rm -rf /tmp/test_env
```

## Troubleshooting

### "Invalid or non-existent authentication information"
- Check your `~/.pypirc` file has correct tokens
- Tokens should start with `pypi-`
- Make sure you're using `__token__` as username

### "File already exists"
- You cannot re-upload the same version
- Increment the version number (e.g., `0.0.1rc1` → `0.0.1rc2`)
- Or delete the release on PyPI/TestPyPI (not recommended)

### "Package name already taken"
- If `ai_energy_benchmarks` is taken, you might need to use a different name
- Try: `ai-energy-benchmarks`, `neuralwatt-ai-benchmarks`, etc.
- Update `name` in `pyproject.toml`

### Dependencies not found when installing from TestPyPI
- Always use `--extra-index-url https://pypi.org/simple/`
- This allows pip to get dependencies from production PyPI

### Import errors after installation
```bash
# Check what's in the wheel
unzip -l dist/ai_energy_benchmarks-*.whl

# Should include:
# - ai_energy_benchmarks/__init__.py
# - ai_energy_benchmarks/py.typed
# - ai_energy_benchmarks/config/reasoning_formats.yaml
```

## Security Best Practices

1. **Never commit tokens to git**
   - Add `.pypirc` to `.gitignore`
   - Use GitHub secrets for automation

2. **Use project-scoped tokens**
   - After first upload, create project-specific tokens
   - Go to project settings → API tokens

3. **Secure your .pypirc**
   ```bash
   chmod 600 ~/.pypirc
   ```

4. **Enable 2FA**
   - Both PyPI and TestPyPI support 2FA
   - Highly recommended for production

## Quick Reference

```bash
# Install publishing tools
pip install twine build

# Build package
python -m build

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    ai_energy_benchmarks

# Install from PyPI
pip install ai_energy_benchmarks

# Check uploaded packages
# TestPyPI: https://test.pypi.org/project/ai-energy-benchmarks/
# PyPI: https://pypi.org/project/ai-energy-benchmarks/
```

## Next Steps: GitHub Actions Automation

Once you're comfortable with manual publishing, see:
- `/home/scott/src/ai_energy_benchmarks/design/pip_deployment_gap_analysis.md`

For automated workflows that:
- Auto-publish to TestPyPI on push to `ppe`
- Auto-publish to PyPI on version tags
- Run tests before publishing
- Create GitHub releases

---

**You're now ready to publish!** Start with TestPyPI to practice, then move to production PyPI when ready.
