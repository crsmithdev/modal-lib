"""A stand-in for a training script: a load phase, then STEPS steps of one second."""

import os
import sys
import time

print("loading")
time.sleep(3)
steps = int(os.environ["STEPS"])
for i in range(1, steps + 1):
    print(f"step {i}/{steps}")
    time.sleep(1)
    if os.environ.get("FAIL") and i == steps:
        print("diverged", file=sys.stderr)
        sys.exit(3)
print("done")
