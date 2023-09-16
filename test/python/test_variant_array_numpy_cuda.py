from teca import *
import numpy as np
import sys

stderr = sys.__stderr__

n_elem = 256
init_val = 3.1415
mod_val = 10000
res_val = init_val*mod_val

# send data from C++ to Python
stderr.write('TEST: C++ --> Python\n' \
             '=======================\n')

stderr.write('TEST: creating a teca_variant_array w. CUDA ... \n')
varr = teca_float_array.New(variant_array_allocator_cuda)
varr.resize(n_elem, init_val)
stderr.write('varr = %s\n'%(str(varr)))
stderr.write('TEST: creating a teca_variant_array w. CUDA ... OK!\n\n')

stderr.write('TEST: get a handle to the data ... \n')
h = varr.get_host_accessible()
stderr.write('TEST: get a handle to the data ... OK!\n\n')

stderr.write('TEST: share the data with Numpy ... \n')
arr = np.array(h, copy=False)
stderr.write('arr.__array_interface__ = %s\n'%(arr.__array_interface__))
stderr.write('handle ref count %d\n'%(sys.getrefcount(h)-1))
stderr.write('TEST: share the data with Numpy ... OK!\n\n')

stderr.write('TEST: deleting the teca_variant_array ... \n')
varr = None
h = None
stderr.write('TEST: deleting the teca_variant_array ... OK!\n\n')

stderr.write('TEST: Numpy reads the data ... \n')
stderr.write('arr = %s\n'%(str(arr)))
stderr.write('TEST: Numpy reads the data ... OK!\n\n')

stderr.write('TEST: Numpy modifies the data ... \n')
arr *= mod_val
stderr.write('arr = %s\n'%(str(arr)))
stderr.write('TEST: Numpy modifies the data ... OK!\n\n')

stderr.write('TEST: Verify the result ... \n')
if not np.allclose(arr, res_val):
    stderr.write('ERROR: TEST failed!\n')
    sys.exit(-1)
stderr.write('TEST: Verify the result ... OK\n\n')

stderr.write('TEST: deleting the Numpy array ... \n')
arr = None
stderr.write('TEST: deleting the Numpy array ... OK!\n\n')

sys.exit(0)
