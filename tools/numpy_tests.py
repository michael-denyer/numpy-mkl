import os
import sys
from pathlib import Path

import numpy as np

np.show_config()

if os.environ['RUNNER_OS'] == 'Windows':
    # GH 20391
    libs = Path(sys.prefix) / 'libs'
    libs.mkdir(parents=True, exist_ok=True)

# Use same memory when running tests as upstream. This means we'll skip
# the same tests due to memory limitations.
os.environ['NPY_AVAILABLE_MEM'] = '4 GB'

# Per-test timeout (60s) to catch hanging tests faster
# ILP64 builds may have tests that hang due to different integer handling
# -n=auto runs tests in parallel using pytest-xdist
# --timeout=60 requires pytest-timeout plugin
extra_args = ['-n=auto', '--timeout=60']

passed = np.test(label='full', extra_argv=extra_args)
sys.exit(not passed)
