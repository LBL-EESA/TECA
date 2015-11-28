from py_teca_core import *
from array import array
md = teca_metadata()
ext = array('i', [0,1,2,3,4,5])
md.declare_i("ext")
md.append_i("ext", ext)
#md.append_i("ext", [7])
print md
