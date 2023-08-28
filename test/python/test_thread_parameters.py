from teca import *
from mpi4py.MPI import COMM_SELF

n,affin,devs = thread_util.thread_parameters(COMM_SELF, -1, 1, -1, -1, 1)

print('num_threads = %d'%(n))
print('affinity = %s'%(str(affin)))
print('device_ids = %s'%(str(devs)))
