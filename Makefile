.PHONY: help style check-whitespace check-commit-msg fix-whitespace fix-whitespace-last-commit format black test clean

# Python executable
PYTHON := python3

# Black formatter options
BLACK_OPTS := --line-length 100

# Files to check/format
PYTHON_FILES := sartop.py scripts/*.py

help:
	@echo "SARTop Makefile targets:"
	@echo ""
	@echo "  style                      - Run all style checks (whitespace + commit msg)"
	@echo "  check-whitespace           - Check for whitespace issues"
	@echo "  check-commit-msg           - Check last commit message format"
	@echo "  fix-whitespace             - Auto-fix whitespace in modified files"
	@echo "  fix-whitespace-last-commit - Fix whitespace in files from last commit"
	@echo "  format                     - Format code with black"
	@echo "  black                      - Alias for format"
	@echo "  test                       - Run basic sanity tests"
	@echo "  clean                      - Remove generated output files"
	@echo ""

# Main style check target - checks both whitespace and commit message
style: check-whitespace check-commit-msg
	@echo ""
	@echo "✓ All style checks passed!"

# Check for whitespace issues in Python files
check-whitespace:
	@echo "Checking for whitespace issues..."
	@$(PYTHON) scripts/check_whitespace.py $(PYTHON_FILES)

# Check commit message format (Generated-by and Signed-off-by)
check-commit-msg:
	@echo "Checking commit message format..."
	@$(PYTHON) scripts/check_commit_msg.py

# Auto-fix whitespace issues in modified files
fix-whitespace:
	@echo "Fixing whitespace issues in modified files..."
	@git diff --name-only --diff-filter=AM | grep '\.py$$' | xargs -r $(PYTHON) scripts/fix_whitespace_issues.py
	@echo "✓ Whitespace fixed in modified files"

# Fix whitespace only in files from the last commit
fix-whitespace-last-commit:
	@echo "Fixing whitespace in files from last commit..."
	@git diff --name-only HEAD~1 HEAD | grep '\.py$$' | xargs -r $(PYTHON) scripts/fix_whitespace_issues.py
	@if git diff --quiet; then \
		echo "✓ No whitespace changes needed"; \
	else \
		echo "✓ Whitespace fixed - review changes and amend commit if needed"; \
		git diff --stat; \
	fi

# Format Python code with black
format:
	@echo "Formatting Python code with black..."
	@if command -v black >/dev/null 2>&1; then \
		black $(BLACK_OPTS) $(PYTHON_FILES); \
		echo "✓ Code formatted with black"; \
	else \
		echo "ERROR: black not installed. Install with: pip install black"; \
		exit 1; \
	fi

# Alias for format
black: format

# Run basic sanity tests
test:
	@echo "Running sanity tests..."
	@$(PYTHON) -m py_compile sartop.py
	@$(PYTHON) sartop.py --help >/dev/null
	@echo "✓ Basic tests passed"

# Clean generated output files
clean:
	@echo "Cleaning generated files..."
	@rm -f sartop-*.json sartop-*_plot.png sartop-*_summary.txt
	@rm -rf __pycache__ *.pyc
	@echo "✓ Cleaned generated files"
