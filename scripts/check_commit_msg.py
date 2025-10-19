#!/usr/bin/env python3
"""Check commit message format for Generated-by and Signed-off-by tags."""

import sys
import subprocess

def check_commit_message():
    """Check the last commit message format."""
    try:
        # Get the last commit message
        result = subprocess.run(
            ['git', 'log', '-1', '--pretty=%B'],
            capture_output=True,
            text=True,
            check=True
        )
        msg = result.stdout.strip()
        lines = msg.split('\n')

        # Find tag lines
        gen_by_idx = None
        signed_off_idx = None

        for i, line in enumerate(lines):
            if 'Generated-by: Claude AI' in line:
                gen_by_idx = i
            if 'Signed-off-by:' in line:
                signed_off_idx = i

        # Check if both tags exist
        if gen_by_idx is not None and signed_off_idx is not None:
            # They must be consecutive (no empty lines between)
            if signed_off_idx - gen_by_idx != 1:
                print("ERROR: Generated-by and Signed-off-by must be consecutive")
                print("       (no empty lines between them)")
                print(f"\nFound at lines: Generated-by={gen_by_idx+1}, Signed-off-by={signed_off_idx+1}")
                return False
            else:
                print("✓ Commit message format is correct")
                return True

        elif gen_by_idx is not None or signed_off_idx is not None:
            print("WARNING: Commit has one tag but not both")
            print("         (Generated-by and Signed-off-by)")
            return True  # Warning, not error

        else:
            print("INFO: No Generated-by or Signed-off-by tags found")
            return True  # Not required for all commits

    except subprocess.CalledProcessError:
        print("ERROR: Could not get commit message (not in a git repo?)")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == '__main__':
    if check_commit_message():
        sys.exit(0)
    else:
        sys.exit(1)
