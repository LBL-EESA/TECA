from teca import *
import cupy as cp
import sys

stderr = sys.__stderr__

n_elem = 256
init_val = 3.1415
mod_val = 10000
res_val = init_val*mod_val

# send data from C++ to Python
stderr.write('TEST: C++ --> Python\n' \
             '=======================\n')

stderr.write('TEST: creating a teca_variant_array CPU ... \n')
varr = teca_float_array.New(variant_array_allocator_malloc)
varr.resize(n_elem, init_val)
stderr.write('varr = %s\n'%(str(varr)))
stderr.write('TEST: creating a teca_variant_array CPU ... OK!\n\n')

stderr.write('TEST: get a handle to the data ... \n')
h = varr.get_cuda_accessible()
stderr.write('TEST: get a handle to the data ... OK!\n\n')

stderr.write('TEST: share the data with Cupy ... \n')
arr = cp.array(h, copy=False)
stderr.write('arr.__cuda_array_interface__ = %s\n'%(arr.__cuda_array_interface__))
stderr.write('handle ref count %d\n'%(sys.getrefcount(h)-1))
stderr.write('TEST: share the data with Cupy ... OK!\n\n')

stderr.write('TEST: deleting the teca_variant_array ... \n')
varr = None
h = None
stderr.write('TEST: deleting the teca_variant_array ... OK!\n\n')

stderr.write('TEST: Cupy reads the data ... \n')
stderr.write('arr = %s\n'%(str(arr)))
stderr.write('TEST: Cupy reads the data ... OK!\n\n')

stderr.write('TEST: Cupy modifies the data ... \n')
arr *= mod_val
stderr.write('arr = %s\n'%(str(arr)))
stderr.write('TEST: Cupy modifies the data ... OK!\n\n')

stderr.write('TEST: Verify the result ... \n')
if not cp.allclose(arr, res_val):
    stderr.write('ERROR: TEST failed!\n')
    sys.exit(-1)
stderr.write('TEST: Verify the result ... OK\n\n')

stderr.write('TEST: deleting the Cupy array ... \n')
arr = None
stderr.write('TEST: deleting the Cupy array ... OK!\n\n')

sys.exit(0)
