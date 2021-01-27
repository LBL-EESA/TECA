import sys
from mpi4py import MPI
from teca import *

n_explicit = 3
verbose = 1

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_ranks = comm.Get_size()

if rank == 0:
    sys.stderr.write('Tetsing on %d ranks, in manual mode request %d threads\n'%(n_ranks, n_explicit))
    sys.stderr.flush()
comm.Barrier()

# auto w/ affinity map
n_threads = 1
affinity = []
if rank == 0:
    sys.stderr.write('\n\nTesting automatic load balancing w/ affiniity map\n')
try:
    n_threads, affinity = thread_util.thread_parameters(MPI.COMM_WORLD, -1, 1, verbose)
except(Exception):
    sys.stderr.write('Failed to determine threading parameters\n')
for i in range(n_ranks):
    if i == rank:
        sys.stderr.write('Rank %d : n_threads = %d, affinity = %s\n'%(rank, n_threads, affinity))
        sys.stderr.flush()
    comm.Barrier()

# auto w/o affinity map
n_threads = 1
affinity = []
if rank == 0:
    sys.stderr.write('\n\nTesting automatic load balancing w/o affiniity map\n')
try:
    n_threads, affinity = thread_util.thread_parameters(MPI.COMM_WORLD, -1, 0, 0)
except(Exception):
    sys.stderr.write('Failed to determine threading parameters\n')
for i in range(n_ranks):
    if i == rank:
        sys.stderr.write('Rank %d : n_threads = %d, affinity = %s\n'%(rank, n_threads, affinity))
        sys.stderr.flush()
    comm.Barrier()

# explicit w/ affinity map
n_threads = 1
affinity = []
if rank == 0:
    sys.stderr.write('\n\nTesting explcit load balancing (%d threads per rank) w/ affiniity map\n'%(n_explicit))
try:
    n_threads,affinity = thread_util.thread_parameters(MPI.COMM_WORLD, n_explicit, 1, verbose)
except(Exception):
    sys.stderr.write('Failed to determine threading parameters\n')
for i in range(n_ranks):
    if i == rank:
        sys.stderr.write('Rank %d : n_threads = %d, affinity = %s\n'%(rank, n_threads, affinity))
        sys.stderr.flush()
    comm.Barrier()

# explicit w/o affinity map
n_threads = 1
affinity = []
if rank == 0:
    sys.stderr.write('\n\nTesting explcit load balancing (%d threads per rank) w/o affiniity map\n'%(n_explicit))
try:
    n_threads, affinity = thread_util.thread_parameters(MPI.COMM_WORLD, n_explicit, 0, 0)
except(Exception):
    sys.stderr.write('Failed to determine threading parameters\n')
for i in range(n_ranks):
    if i == rank:
        sys.stderr.write('Rank %d : n_threads = %d, affinity = %s\n'%(rank, n_threads, affinity))
        sys.stderr.flush()
    comm.Barrier()
