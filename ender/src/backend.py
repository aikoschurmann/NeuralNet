import os
import sys

# Set ENDER_USE_GPU=1 in your environment variables to enable CuPy
USE_GPU = os.environ.get('ENDER_USE_GPU', '0') == '1'

if USE_GPU:
    try:
        import cupy as np
        print("ender backend: Using CuPy (GPU Accelerated)")
    except ImportError:
        print("ender backend: WARNING - ENDER_USE_GPU is set, but 'cupy' is not installed. Falling back to CPU.")
        import numpy as np
else:
    import numpy as np

# Export np so other modules can import it seamlessly
sys.modules[__name__].np = np
