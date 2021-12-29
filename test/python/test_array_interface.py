from teca import *
import numpy as np
import sys
import gc

err = sys.__stderr__



a = teca_double_array.New()
a.resize(16, 3.1415)
#sa = a.__array_struct__

err.write('teca_variant_array\n')
err.write('a = %s\n'%(str(a)))

b = np.array(a, copy=False)
#sb = b.__array_interface__

err.write('numpy array via __array_struct__\n')
err.write('b = %s\n'%(str(b)))
#err.write('sb = %s\n'%(str(sb)))


del a
a = None
#sa = None
#err.write('deleted the teca_variant_array\n')

b = b.reshape((4,4))
#sb = b.__array_interface__

err.write('numpy array after reshape\n')
err.write('b = %s\n'%(str(b)))
#err.write('sb = %s\n'%(str(sb)))

b[:,:] = -1.0

err.write('numpy array after set to -1\n')
err.write('b = %s\n'%(str(b)))
#err.write('sb = %s\n'%(str(sb)))


b = None
#sb = None
err.write('deleted the numpy array\n')


#err.write('teca_variant_array\n')
#err.write('a = %s\n'%(str(a)))

