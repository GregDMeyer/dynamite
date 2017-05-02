
_initialized = False

import slepc4py

def initialize(slepc_args=[]):
    global _initialized
    if not _initialized:
        slepc4py.init(slepc_args)
        _initialized = True
