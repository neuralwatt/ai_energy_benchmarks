# GitHub Actions PyPI Publishing - Summary

## What Was Created

### 1. Main Workflow File
**Location**: `.github/workflows/pypi-publish.yml`

This GitHub Actions workflow automates the entire PyPI publishing process with the following features:

#### Features:
- ✅ **Automatic Testing**: Runs linting, type checking, and tests before publishing
- ✅ **Branch-Based Publishing**:
  - `ppe` branch → TestPyPI (https://test.pypi.org)
  - `main` branch → Production PyPI (https://pypi.org)
- ✅ **Trusted Publishing**: Uses OpenID Connect (OIDC) - no API tokens needed!
- ✅ **Build Artifacts**: Saves build artifacts for debugging
- ✅ **GitHub Releases**: Automatically creates releases for production
- ✅ **Smart Triggers**: Only runs when package files change
- ✅ **Manual Triggers**: Can be manually triggered via GitHub UI

#### Workflow Jobs:

1. **Test Job** (runs on all branches):
   - Linting with `ruff`
   - Type checking with `mypy`
   - Unit tests with `pytest`

2. **Build Job** (runs after tests pass):
   - Builds wheel and source distribution
   - Validates with `twine check`
   - Uploads artifacts

3. **Publish Job** (branch-dependent):
   - PPE → TestPyPI
   - Main → Production PyPI
   - Creates helpful summary with installation instructions

4. **Release Job** (main branch only):
   - Creates Git tag (e.g., `v0.0.4`)
   - Creates GitHub release

### 2. Setup Guide
**Location**: `.github/PYPI_SETUP.md`

Comprehensive guide covering:
- ✅ Trusted Publishing setup (recommended)
- ✅ Alternative API token setup
- ✅ GitHub Environment configuration
- ✅ Version management strategies
- ✅ Publishing workflow for both branches
- ✅ Troubleshooting common issues
- ✅ Security best practices

## Quick Setup Checklist

### Prerequisites
- [ ] PyPI account created at https://pypi.org
- [ ] TestPyPI account created at https://test.pypi.org

### On PyPI/TestPyPI (Trusted Publishing - Recommended)
- [ ] Configure TestPyPI trusted publisher:
  - Project name: `ai-energy-benchmarks`
  - Repository: `neuralwatt/ai_energy_benchmarks`
  - Workflow: `pypi-publish.yml`
  - Environment: `testpypi`
- [ ] Configure PyPI trusted publisher:
  - Project name: `ai-energy-benchmarks`
  - Repository: `neuralwatt/ai_energy_benchmarks`
  - Workflow: `pypi-publish.yml`
  - Environment: `pypi`

### On GitHub
- [ ] Create `testpypi` environment in repository settings
  - Optional: Add branch protection for `ppe` only
- [ ] Create `pypi` environment in repository settings
  - **Recommended**: Add required reviewers (prevent accidental deploys)
  - Recommended: Add branch protection for `main` only

## How to Use

### Testing on PPE Branch

```bash
# 1. Switch to PPE branch
git checkout ppe
git pull origin ppe

# 2. Update version for testing
echo "0.0.4rc1" > VERSION.txt
# Update version in pyproject.toml

# 3. Make your changes
# ... edit code ...

# 4. Commit and push
git add .
git commit -m "Test release candidate 0.0.4rc1"
git push origin ppe

# 5. GitHub Actions will automatically:
#    - Run tests
#    - Build package
#    - Publish to TestPyPI
#    - Show installation instructions

# 6. Test the installation
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    ai_energy_benchmarks==0.0.4rc1
```

### Production Release on Main Branch

```bash
# 1. After successful PPE testing, switch to main
git checkout main
git pull origin main

# 2. Update version for production
echo "0.0.4" > VERSION.txt
# Update version in pyproject.toml to "0.0.4"

# 3. Commit and push
git add VERSION.txt pyproject.toml
git commit -m "Release version 0.0.4"
git push origin main

# 4. GitHub Actions will:
#    - Run tests
#    - Build package
#    - Wait for approval (if environment protection enabled)
#    - Publish to PyPI
#    - Create GitHub release with tag v0.0.4

# 5. Verify installation
pip install ai_energy_benchmarks==0.0.4
```

## Key Differences from Manual Process

### Before (Manual):
1. Run tests locally
2. Build package locally
3. Configure `~/.pypirc` with tokens
4. Upload with `twine upload`
5. Manually create git tags
6. Manually create GitHub releases

### After (Automated):
1. Push to `ppe` or `main` branch
2. GitHub Actions handles everything
3. No local tokens needed (trusted publishing)
4. Automatic releases and tags
5. Built-in testing before publishing
6. Can't accidentally publish broken code

## Security Improvements

### Trusted Publishing Benefits:
- ✅ No API tokens to manage
- ✅ No secrets in GitHub
- ✅ Short-lived credentials (OIDC)
- ✅ Scoped to specific workflow
- ✅ Can't be leaked or stolen

### Environment Protection:
- ✅ Require manual approval for production
- ✅ Restrict to specific branches
- ✅ Audit trail of who approved deployments
- ✅ Prevent accidental publishes

## Version Strategy

### TestPyPI (PPE Branch)
Use pre-release versions:
- Development: `0.0.4.dev1`, `0.0.4.dev2`, ...
- Release candidates: `0.0.4rc1`, `0.0.4rc2`, ...
- Alpha/Beta: `0.0.4a1`, `0.0.4b1`, ...

### PyPI (Main Branch)
Use semantic versioning:
- Patch (bug fixes): `0.0.3` → `0.0.4`
- Minor (new features): `0.0.4` → `0.1.0`
- Major (breaking changes): `0.1.0` → `1.0.0`

## Workflow Triggers

The workflow runs when:
- ✅ Push to `ppe` or `main` branch
- ✅ Changes to these paths:
  - `ai_energy_benchmarks/**`
  - `pyproject.toml`
  - `VERSION.txt`
  - `setup.py`
  - `MANIFEST.in`
- ✅ Manual trigger via Actions UI

## Monitoring

### View Workflow Runs
- GitHub Actions tab: https://github.com/neuralwatt/ai_energy_benchmarks/actions

### View Published Packages
- TestPyPI: https://test.pypi.org/project/ai-energy-benchmarks/
- PyPI: https://pypi.org/project/ai-energy-benchmarks/

### View Releases
- GitHub Releases: https://github.com/neuralwatt/ai_energy_benchmarks/releases

## Common Scenarios

### Scenario 1: Testing a New Feature
```bash
git checkout ppe
# ... develop feature ...
echo "0.0.5rc1" > VERSION.txt
git commit -m "Test new feature"
git push origin ppe
# Wait for TestPyPI publish
# Test installation
```

### Scenario 2: Production Release
```bash
git checkout main
echo "0.0.5" > VERSION.txt
git commit -m "Release 0.0.5"
git push origin main
# Approve in Actions UI (if required)
# Automatic PyPI publish and GitHub release
```

### Scenario 3: Hotfix
```bash
git checkout main
# ... fix bug ...
echo "0.0.6" > VERSION.txt
git commit -m "Hotfix: critical bug"
git push origin main
# Immediate publish after approval
```

### Scenario 4: Manual Trigger
1. Go to Actions tab
2. Select "Publish to PyPI"
3. Click "Run workflow"
4. Choose branch (`ppe` or `main`)
5. Click "Run workflow" button

## Troubleshooting Quick Reference

| Error | Solution |
|-------|----------|
| "File already exists" | Increment version number |
| "Trusted publisher not configured" | Set up on PyPI/TestPyPI |
| Tests failing | Run tests locally first |
| Workflow not triggering | Check file paths or trigger manually |
| Environment approval stuck | Go to Actions and approve |

## Next Steps

1. **Setup**: Follow `.github/PYPI_SETUP.md` to configure trusted publishing
2. **Test**: Push to `ppe` branch to test the workflow
3. **Review**: Check the Actions tab to see the workflow run
4. **Verify**: Install from TestPyPI to verify it works
5. **Deploy**: When ready, push to `main` for production release

## Files Created

```
ai_energy_benchmarks/
├── .github/
│   ├── workflows/
│   │   └── pypi-publish.yml          # Main workflow file
│   └── PYPI_SETUP.md                 # Setup guide
└── ai_helpers/
    └── github_actions_pypi_summary.md # This file
```

## Additional Resources

- GitHub Actions docs: https://docs.github.com/en/actions
- PyPI Trusted Publishing: https://docs.pypi.org/trusted-publishers/
- PyPA publish action: https://github.com/pypa/gh-action-pypi-publish

---

**Ready to go!** Follow the setup guide in `.github/PYPI_SETUP.md` to configure trusted publishing.
