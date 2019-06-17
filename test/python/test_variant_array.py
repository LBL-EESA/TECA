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

nptypes = [np.float32, np.float64, np.int8, np.int16, \
    np.int32, np.int64, np.uint8, np.uint16, np.uint32, \
    np.uint64, np.long, np.byte]

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
    arr.resize(20 + len(nptypes))
    q = 0
    for i in range(10):
        arr[q] = int(i)
        q += 1
    for i in range(10):
        arr[q] = float(10 - i)
        q += 1
    for t in nptypes:
        arr[q] = t(q)
        q += 1
end_sec('set')

start_sec('append')
for arr in arrs:
    out('%s\n'%(str(type(arr))))
    arr.append(42)
    arr.append(3.14)
    arr.append([70, 80, 90])
    q = 0
    for t in nptypes:
        arr.append(t(q))
        q += 1
    for t in nptypes:
        v = np.ones(3, dtype=t)*q
        arr.append(v)
        q += 1
end_sec('append')

start_sec('get')
for arr in arrs:
    out('%s\n'%(str(type(arr))))
    for i in range(20):
        out('%g '%(arr[i]))
    out('\n')
end_sec('get')

start_sec('bounds check')
for arr in arrs:
    try:
        arr.set(3000, 1.0)
    except:
        out('%s .. caught set out of bounds!\n'%(str(type(arr))))
    try:
        arr.get(3000)
    except:
        out('%s .. caught get out of bounds!\n'%(str(type(arr))))
end_sec('bounds check')

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
