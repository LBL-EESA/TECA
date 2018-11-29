from teca import *
import numpy as np
import sys

set_stack_trace_on_error()

def out(s):
    sys.stderr.write(s)

def start_sec(s):
    out('checking '+s+'...\n')

def end_sec(s):
    out('checking '+s+'...ok\n\n')


start_sec('constructor')
arrs = []
arrs.append(teca_double_array.New())
arrs.append(teca_float_array.New())
arrs.append(teca_char_array.New())
arrs.append(teca_int_array.New())
arrs.append(teca_long_long_array.New())
end_sec('constructor')

start_sec('set')
for arr in arrs:
    out('%s\n'%(str(type(arr))))
    arr.resize(20)
    for i in range(10):
        arr[i] = int(i)
    for i in range(10):
        arr[i+10] = float(10 - i)
end_sec('set')

start_sec('append')
for arr in arrs:
    out('%s\n'%(str(type(arr))))
    arr.append(42)
    arr.append(3.14)
    arr.append([70, 80, 90])
    arr.append(np.array([3,2,1], dtype=np.float64))
end_sec('append')

start_sec('get')
for arr in arrs:
    out('%s\n'%(str(type(arr))))
    for i in range(20):
        out('%g '%(arr[i]))
    out('\n')
end_sec('get')

start_sec('as_array')
for arr in arrs:
    out('%s\n'%(str(type(arr))))
    out('%s\n'%(str(arr.as_array())))
end_sec('as_array')

start_sec('Python object constructor')
for arr in arrs:
    out('%s'%(str(type(arr))))
    a = arr.as_array()
    b = teca_variant_array.New(a)
    out(' -> %s -> %s\n'%(str(type(a)), str(type(b))))
    out('    %s\n'%(str(arr)))
end_sec('Python object constructor')

start_sec('str')
for arr in arrs:
    out('%s\n'%(str(type(arr))))
    out('%s\n'%(str(arr)))
end_sec('str')

start_sec('iter')
for arr in arrs:
    out('%s\n  '%(str(type(arr))))
    for v in arr:
        out('%s '%(v))
    out('\n')
end_sec('iter')

start_sec('destructor')
for arr in arrs:
    out('%s\n'%(str(type(arr))))
    del arr
end_sec('destructor')
