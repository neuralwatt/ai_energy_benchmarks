# GitHub Actions PyPI Publishing Setup Guide

This guide explains how to configure the GitHub Actions workflow for automated PyPI publishing.

## Overview

The workflow automatically publishes the package:
- **PPE branch** → TestPyPI (https://test.pypi.org)
- **Main branch** → Production PyPI (https://pypi.org)

## Trusted Publishing (Recommended)

We use PyPI's **Trusted Publishing** feature, which uses OpenID Connect (OIDC) instead of API tokens. This is more secure and doesn't require managing secrets.

### Step 1: Configure TestPyPI (for PPE branch)

1. **Create TestPyPI account** (if you don't have one):
   - Go to https://test.pypi.org/account/register/
   - Verify your email

2. **Add Trusted Publisher on TestPyPI**:
   - Go to https://test.pypi.org/manage/account/publishing/
   - Click "Add a new pending publisher"
   - Fill in the form:
     - **PyPI Project Name**: `ai-energy-benchmarks`
     - **Owner**: `neuralwatt` (your GitHub username/org)
     - **Repository name**: `ai_energy_benchmarks`
     - **Workflow name**: `pypi-publish.yml`
     - **Environment name**: `testpypi`
   - Click "Add"

### Step 2: Configure Production PyPI (for main branch)

1. **Create PyPI account** (if you don't have one):
   - Go to https://pypi.org/account/register/
   - Verify your email

2. **Add Trusted Publisher on PyPI**:
   - Go to https://pypi.org/manage/account/publishing/
   - Click "Add a new pending publisher"
   - Fill in the form:
     - **PyPI Project Name**: `ai-energy-benchmarks`
     - **Owner**: `neuralwatt` (your GitHub username/org)
     - **Repository name**: `ai_energy_benchmarks`
     - **Workflow name**: `pypi-publish.yml`
     - **Environment name**: `pypi`
   - Click "Add"

### Step 3: Configure GitHub Environments

1. **Go to your GitHub repository**:
   - Navigate to `Settings` → `Environments`

2. **Create `testpypi` environment**:
   - Click "New environment"
   - Name: `testpypi`
   - (Optional) Add protection rules:
     - Required reviewers: Add yourself or team members
     - Deployment branches: `ppe` branch only
   - Click "Configure environment"

3. **Create `pypi` environment**:
   - Click "New environment"
   - Name: `pypi`
   - (Optional but RECOMMENDED) Add protection rules:
     - ✅ **Required reviewers**: Add yourself (prevents accidental publishing)
     - Deployment branches: `main` branch only
   - Click "Configure environment"

## Alternative: API Token Method (Legacy)

If you prefer to use API tokens instead of trusted publishing:

### TestPyPI Token Setup

1. Generate TestPyPI token:
   - Go to https://test.pypi.org/manage/account/token/
   - Click "Add API token"
   - Token name: `github_actions_testpypi`
   - Scope: "Entire account"
   - Copy the token (starts with `pypi-`)

2. Add to GitHub Secrets:
   - Go to repository `Settings` → `Secrets and variables` → `Actions`
   - Click "New repository secret"
   - Name: `TEST_PYPI_API_TOKEN`
   - Value: Paste the TestPyPI token
   - Click "Add secret"

3. Modify workflow file (`.github/workflows/pypi-publish.yml`):
   ```yaml
   # In publish-testpypi job, replace:
   permissions:
     id-token: write

   # With:
   # (remove permissions block)

   # And modify the publish step:
   - name: Publish to TestPyPI
     uses: pypa/gh-action-pypi-publish@release/v1
     with:
       repository-url: https://test.pypi.org/legacy/
       password: ${{ secrets.TEST_PYPI_API_TOKEN }}
   ```

### Production PyPI Token Setup

1. Generate PyPI token:
   - Go to https://pypi.org/manage/account/token/
   - Click "Add API token"
   - Token name: `github_actions_pypi`
   - Scope: "Entire account"
   - Copy the token

2. Add to GitHub Secrets:
   - Name: `PYPI_API_TOKEN`
   - Value: Paste the PyPI token

3. Modify workflow file:
   ```yaml
   # In publish-pypi job, modify similar to TestPyPI above
   - name: Publish to PyPI
     uses: pypa/gh-action-pypi-publish@release/v1
     with:
       password: ${{ secrets.PYPI_API_TOKEN }}
   ```

## How It Works

### Workflow Triggers

The workflow runs automatically when:
- Code is pushed to `ppe` or `main` branch
- Changes affect package files:
  - `ai_energy_benchmarks/**`
  - `pyproject.toml`
  - `VERSION.txt`
  - `setup.py`
  - `MANIFEST.in`
- Manually triggered via GitHub Actions UI

### Workflow Steps

1. **Test Job** (runs for both branches):
   - Checkout code
   - Set up Python 3.10
   - Install dependencies
   - Run linting (ruff)
   - Run type checking (mypy)
   - Run tests (pytest)

2. **Build Job** (runs after tests pass):
   - Build Python package (wheel + source distribution)
   - Check package with twine
   - Upload build artifacts

3. **Publish Job** (branch-dependent):
   - **PPE branch**: Publish to TestPyPI
   - **Main branch**: Publish to production PyPI
   - Create installation instructions in job summary

4. **GitHub Release** (main branch only):
   - Create Git tag (e.g., `v0.0.4`)
   - Create GitHub release with version notes

## Version Management

### For PPE Branch (TestPyPI)

Use pre-release version numbers:
```bash
# Development versions
echo "0.0.4.dev1" > VERSION.txt

# Release candidates
echo "0.0.4rc1" > VERSION.txt

# Alpha/Beta
echo "0.0.4a1" > VERSION.txt
echo "0.0.4b1" > VERSION.txt
```

Also update `pyproject.toml`:
```toml
version = "0.0.4rc1"
```

### For Main Branch (Production PyPI)

Use semantic versioning:
```bash
# Patch release (bug fixes)
echo "0.0.4" > VERSION.txt

# Minor release (new features)
echo "0.1.0" > VERSION.txt

# Major release (breaking changes)
echo "1.0.0" > VERSION.txt
```

**Important**: PyPI does not allow re-uploading the same version. Always increment the version number!

## Publishing Workflow

### Testing with PPE Branch

1. **Create/switch to PPE branch**:
   ```bash
   git checkout ppe
   git pull origin ppe
   ```

2. **Make changes and update version**:
   ```bash
   # Edit code
   echo "0.0.4rc1" > VERSION.txt
   # Update version in pyproject.toml
   ```

3. **Commit and push**:
   ```bash
   git add .
   git commit -m "Update for testing: version 0.0.4rc1"
   git push origin ppe
   ```

4. **Monitor GitHub Actions**:
   - Go to repository → "Actions" tab
   - Watch the workflow run
   - If it fails, check the logs
   - If environment protection is enabled, approve the deployment

5. **Test installation**:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/ \
       ai_energy_benchmarks==0.0.4rc1
   ```

### Production Release with Main Branch

1. **Ensure PPE testing is successful**

2. **Switch to main branch**:
   ```bash
   git checkout main
   git pull origin main
   ```

3. **Update version to production**:
   ```bash
   echo "0.0.4" > VERSION.txt
   # Update version in pyproject.toml to "0.0.4"
   ```

4. **Commit and push**:
   ```bash
   git add VERSION.txt pyproject.toml
   git commit -m "Release version 0.0.4"
   git push origin main
   ```

5. **Monitor and approve**:
   - Go to "Actions" tab
   - If environment protection is enabled, **review and approve**
   - Wait for workflow to complete

6. **Verify publication**:
   - Check PyPI: https://pypi.org/project/ai-energy-benchmarks/
   - Check GitHub Releases: https://github.com/neuralwatt/ai_energy_benchmarks/releases
   - Test installation:
     ```bash
     pip install ai_energy_benchmarks==0.0.4
     ```

## Manual Trigger

You can manually trigger the workflow:
1. Go to repository → "Actions" tab
2. Click "Publish to PyPI" workflow
3. Click "Run workflow"
4. Select branch (`ppe` or `main`)
5. Click "Run workflow"

## Troubleshooting

### "File already exists" error
- **Cause**: You're trying to upload a version that already exists
- **Solution**: Increment the version number in `VERSION.txt` and `pyproject.toml`

### "Project name not found" on first upload
- **Cause**: Package name doesn't exist on PyPI yet
- **Solution**:
  - For trusted publishing: The first upload will automatically create the project
  - For API tokens: First upload might fail, then create project-scoped token

### "Trusted publisher not configured"
- **Cause**: Trusted publishing not set up on PyPI/TestPyPI
- **Solution**: Follow "Step 1" and "Step 2" above to configure trusted publishers

### Tests failing
- **Cause**: Code doesn't pass linting, type checking, or tests
- **Solution**: Run locally before pushing:
  ```bash
  ruff check .
  ruff format --check .
  mypy ai_energy_benchmarks/
  pytest -v -m "not integration and not e2e"
  ```

### Workflow not triggering
- **Cause**: Changes don't affect watched paths
- **Solution**: Either modify watched files or manually trigger workflow

### Environment approval stuck
- **Cause**: Environment protection requires manual approval
- **Solution**: Go to Actions tab → click the workflow run → approve deployment

## Security Best Practices

1. **Use Trusted Publishing** (OIDC) instead of API tokens when possible
2. **Enable environment protection** for production PyPI:
   - Require manual approval for deployments
   - Restrict to `main` branch only
3. **Never commit tokens** to the repository
4. **Enable 2FA** on PyPI and TestPyPI accounts
5. **Use project-scoped tokens** after first upload (if using tokens)
6. **Review changes** before approving production deployments

## Monitoring

### Check Workflow Status
- GitHub Actions tab: https://github.com/neuralwatt/ai_energy_benchmarks/actions
- Email notifications: Configure in GitHub Settings → Notifications

### Check Package Status
- TestPyPI: https://test.pypi.org/project/ai-energy-benchmarks/
- PyPI: https://pypi.org/project/ai-energy-benchmarks/

### View Installation Stats
- PyPI downloads: https://pypistats.org/packages/ai-energy-benchmarks

## Next Steps

1. Set up trusted publishing on TestPyPI and PyPI (most important!)
2. Configure GitHub environments with protection rules
3. Test the workflow on PPE branch
4. Once successful, use for production releases on main branch
5. Monitor for any issues and iterate

---

**Questions or issues?** Check the workflow run logs in the Actions tab or open an issue.
