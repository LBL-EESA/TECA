from teca import *
import cupy as cp
import sys

stderr = sys.__stderr__

n_elem = 256
init_val = 3.1415
mod_val = 10000
res_val = init_val*mod_val

# send data from C++ to Python
stderr.write('TEST 1 : C++ --> Python\n' \
             '=======================\n')

stderr.write('TEST 1: creating a teca_variant_array w. CUDA ... \n')
varr = teca_float_array.New(variant_array_allocator_cuda)
varr.resize(n_elem, init_val)
stderr.write('varr = %s\n'%(str(varr)))
stderr.write('TEST 1: creating a teca_variant_array w. CUDA ... OK!\n\n')

stderr.write('TEST 1: get a handle to the data ... \n')
h = varr.get_cuda_accessible()
stderr.write('TEST 1: get a handle to the data ... OK!\n\n')

stderr.write('TEST 1: share the data with Cupy ... \n')
arr = cp.array(h, copy=False)
stderr.write('arr.__cuda_array_interface__ = %s\n'%(arr.__cuda_array_interface__))
stderr.write('handle ref count %d\n'%(sys.getrefcount(h)-1))
stderr.write('TEST 1: share the data with Cupy ... OK!\n\n')

stderr.write('TEST 1: deleting the teca_variant_array ... \n')
varr = None
h = None
stderr.write('TEST 1: deleting the teca_variant_array ... OK!\n\n')

stderr.write('TEST 1: Cupy modifies the data ... \n')
arr *= mod_val
stderr.write('arr = %s\n'%(str(arr)))
stderr.write('TEST 1: Cupy modifies the data ... OK!\n\n')

stderr.write('TEST 1: Verify the result ... \n')
if not cp.allclose(arr, res_val):
    stderr.write('ERROR: TEST 1 failed!\n')
    sys.exit(-1)
stderr.write('TEST 1: Verify the result ... OK\n\n')

stderr.write('TEST 1: deleting the Cupy array ... \n')
arr = None
stderr.write('TEST 1: deleting the Cupy array ... OK!\n\n')



# send data from Python to C++
stderr.write('TEST 2 : Python --> C++\n' \
             '=======================\n')

stderr.write('TEST 2: creating a Cupy array ... \n')
arr = cp.full((n_elem), init_val, dtype='float32')
stderr.write('arr.__cuda_array_interface__ = %s\n'%(arr.__cuda_array_interface__))
#stderr.write('arr = %s\n'%(str(arr)))
stderr.write('TEST 2: creating a Cupy array ... OK\n\n')

stderr.write('TEST 2: share the data with teca_variant_array ... \n')
varr = teca_variant_array.New(arr)
stderr.write('varr = %s\n'%(str(varr)))
stderr.write('TEST 2: share the data with teca_variant_array ... OK\n\n')

stderr.write('TEST 2: Cupy modifies the data ... \n')
arr *= mod_val
#stderr.write('arr = %s\n'%(str(arr)))
stderr.write('TEST 2: Cupy modifies the data ... OK!\n\n')

stderr.write('TEST 2: deleting the Cupy array ... \n')
arr = None
stderr.write('TEST 2: deleting the Cupy array ... OK!\n\n')

stderr.write('TEST 2: display the modified teca_variant_array ... \n')
stderr.write('varr = %s\n'%(str(varr)))
stderr.write('TEST 2: display the modified teca_variant_array ... OK\n\n')

stderr.write('TEST 2: deleting the teca_variant_array ... \n')
varr = None
stderr.write('TEST 2: deleting the teca_variant_array ... OK!\n\n')

sys.exit(0)
