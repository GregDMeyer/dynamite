'''
This files demonstrates an alternative way to coordinate the randomness used by 
each MPI rank.

Try running it like `mpirun -n 3 bcast_seed.py`. You will see the same "random" number
repeated 3 times, once from each MPI rank.
'''

from os import urandom  # hardware random number generator, for picking a seed
import random
from dynamite import config

def set_shared_seed():
    # dynamite must be initialized before importing PETSc
    config.initialize()
    
    from petsc4py import PETSc
    comm = PETSc.COMM_WORLD.tompi4py()  # get the MPI communicator as an mpi4py object

    # have rank 0 pick a seed
    if comm.rank == 0:
        seed = urandom(4)  # 4 hardware random bytes
    else:
        seed = None
    
    # now broadcast from rank 0 to all other ranks
    seed = comm.bcast(seed, root=0)

    # finally seed the random number generator
    random.seed(seed)

# just to show all ranks give the same random number
# try running this script with N MPI ranks, and you will see the same random number
# appear N times!
set_shared_seed()
print(random.random())