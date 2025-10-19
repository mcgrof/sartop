#!/usr/bin/env python3
"""Check for whitespace issues in Python files."""

import sys
import glob

def check_file(filepath):
    """Check a single file for whitespace issues."""
    errors = []
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines, 1):
                # Check for trailing whitespace (but not just \n)
                if line.rstrip('\n').rstrip() != line.rstrip('\n'):
                    errors.append(f"{filepath}:{i}: trailing whitespace")
                # Check for mixed tabs and spaces (indentation)
                if '\t' in line and '    ' in line:
                    stripped = line.lstrip()
                    if stripped and stripped != line:  # Has indentation
                        errors.append(f"{filepath}:{i}: mixed tabs and spaces")

            # Check for newline at EOF
            if lines and not lines[-1].endswith('\n'):
                errors.append(f"{filepath}: missing newline at end of file")

    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return []

    return errors

def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        # Default to checking Python files
        files = ['sartop.py'] + glob.glob('scripts/*.py')

    all_errors = []
    for filepath in files:
        errors = check_file(filepath)
        all_errors.extend(errors)

    if all_errors:
        for error in all_errors:
            print(error)
        sys.exit(1)
    else:
        print("✓ No whitespace issues found")
        sys.exit(0)

if __name__ == '__main__':
    main()
