# Contributing to llm-poe

Thank you for your interest in contributing to llm-poe! This document provides guidelines and information for contributors.

## Development Setup

To set up for development:

```bash
git clone https://github.com/mrf/llm-poe
cd llm-poe
llm install -e .
```

**Important:** Use `llm install -e .` (not `pip install -e .`) for local development to ensure proper plugin discovery.

### Installing Test Dependencies

```bash
pip install -e ".[test]"
```

## Running Tests

This plugin has a comprehensive test suite with 94% code coverage. See [tests/README.md](tests/README.md) for detailed testing documentation.

### Quick Test Commands

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=llm_poe --cov-report=term-missing --cov-report=html

# Run specific test file
pytest tests/test_text_models.py

# View HTML coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Test Coverage Goals

- Aim for >90% code coverage for all new code
- 100% coverage for critical paths (API calls, model registration)
- Test both success and failure cases
- Include edge cases and error conditions

## Code Quality

Before submitting a pull request, ensure your code passes all quality checks:

```bash
# Code formatting (automatically formats code)
black llm_poe.py tests/

# Import sorting
isort llm_poe.py tests/

# Linting
ruff check llm_poe.py tests/
```

These checks run automatically in CI/CD.

## GitHub Actions Workflows

### Test Workflow

**Triggers:** Push to main, Pull Requests

**What it does:**
- Runs 136 tests across multiple platforms and Python versions
- Generates coverage reports (XML and HTML)
- Posts coverage as PR comment (on pull requests)
- Archives HTML coverage reports as artifacts
- Runs code quality checks (Black, isort, ruff)

**Matrix Testing:**
- **Platforms:** Ubuntu, macOS, Windows
- **Python Versions:** 3.8, 3.9, 3.10, 3.11, 3.12
- **Total Combinations:** 15 test runs per push

### Coverage Badge Workflow

**Triggers:** Push to main, Manual dispatch

**What it does:**
- Runs tests with coverage
- Extracts coverage percentage from XML report
- Updates README.md badge with latest coverage
- Automatically commits and pushes changes
- Badge color updates based on coverage level:
  - Green (≥80%)
  - Yellow (60-79%)
  - Orange (40-59%)
  - Red (<40%)

## Setting Up GitHub Integration

If you're maintaining a fork or setting up the workflows in a new repository:

### Step 1: Enable GitHub Actions

GitHub Actions should be enabled by default. Verify in:
`Settings → Actions → General → Allow all actions`

### Step 2: Configure Workflow Permissions

Go to `Settings → Actions → General → Workflow permissions`:
- ✅ Read and write permissions
- ✅ Allow GitHub Actions to create and approve pull requests

This allows the badge workflow to commit README updates.

### Step 3: Configure Branch Protection (Recommended)

Go to `Settings → Branches → Branch protection rules` and add rule for `main`:

- ✅ Require status checks to pass before merging
  - ✅ test (ubuntu-latest, 3.11)
  - ✅ lint
- ✅ Require branches to be up to date before merging

## Pull Request Workflow

When you create a pull request:

1. **Automatic Testing**
   - Tests run on all platforms and Python versions
   - Results appear in PR checks section

2. **Coverage Comment**
   - Bot posts coverage percentage
   - Updates automatically on new commits
   - Links to detailed reports

3. **Status Checks**
   - PR cannot be merged until tests pass
   - Code quality checks must pass

## Viewing Coverage Reports

### In GitHub Actions

1. Go to `Actions` tab
2. Click on a workflow run
3. Scroll to "Artifacts" section
4. Download "coverage-html-report"
5. Extract and open `index.html` in a browser

### In Pull Requests

Coverage is automatically commented on PRs with:
- Coverage percentage
- Link to detailed HTML report in artifacts

## Troubleshooting

### Tests Failing in CI but Passing Locally

- Check Python version compatibility
- Verify all dependencies are in `pyproject.toml`
- Check for platform-specific issues

### Coverage Not Showing in PR Comment

- Check workflow logs for errors in "Create coverage summary" step
- Verify workflow has pull request write permissions
- Ensure coverage.xml is being generated correctly

### Badge Not Updating

- Check `.github/workflows/badge.yml` runs successfully
- Verify workflow has write permissions
- Clear browser cache to see updated badge

## Making Contributions

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and ensure they pass
5. Run code quality checks (black, isort, ruff)
6. Commit your changes with descriptive messages
7. Push to your fork
8. Open a Pull Request

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 License.
