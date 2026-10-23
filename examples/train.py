"""A stand-in for a training script: a load phase, then STEPS steps of one second.

FAIL makes the last step exit 3. LOSE=n stops the container at step n of the first attempt.
"""

import os
import sys
import time

print("loading")
time.sleep(3)
steps = int(os.environ["STEPS"])
for i in range(1, steps + 1):
    print(f"step {i}/{steps}")
    time.sleep(1)
    if os.environ.get("LOSE") == str(i) and os.environ["MLIB_ATTEMPT"] == "0":
        os.kill(1, 9)  # stop the whole container, as a lost machine would
    if os.environ.get("FAIL") and i == steps:
        print("diverged", file=sys.stderr)
        sys.exit(3)
print("done")
