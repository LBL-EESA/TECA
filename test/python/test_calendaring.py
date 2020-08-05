from teca import *
import numpy as np
import sys

# this tests the calendaring utilities and exercises numpy scalar conversion
# typemaps. define TECA_DEBUG_TYPES in teca_py_array.i for addtional debugging
# output.

def status(msg):
    sys.stderr.write('%s\n'%(msg))
    sys.stderr.flush()

units = 'Hours since 2020-08-10 00:00:00'
calendar = 'standard'
t_str = '2020-08-10 12:00:00'

status('tests will used units of %s in the %s calendar'%(units, calendar))

# test numpy and python floating point types. this function can
# only operate on floating point data
ttests = [np.float32, np.float64]
for ttest in ttests:
    t = np.linspace(0, 23, num=24, dtype=ttest)
    status('testing time_step_of on type %s = %s ... '%(str(type(t[0])), str(t)))
    step = coordinate_util.time_step_of(t, True, True, calendar, units, t_str)
    status('time step of %s is -> %d'%(t_str, step))
    if step == 12:
        status('OK!')
    else:
        status('ERROR')

t = [float(v) for v in range(0, 24)]
status('testing time_step_of on type %s = %s ... '%(str(type(t[0])), str(t)))
step = coordinate_util.time_step_of(t, True, True, calendar, units, t_str)
status('time step of %s is -> %d'%(t_str, step))
if step == 12:
    status('OK!')
else:
    status('ERROR')

# test passing numpy scalars
offs = [np.float32(19), np.float64(19), np.int8(19),
        np.int16(19), np.int32(19), np.int64(19),
        int(19), float(19)]

for off in offs:
    status('testing date on type %s = %s ... '%(str(type(off)), str(off)))

    y,mo,d,h,mn,s = calendar_util.date(off, units, calendar)

    status('%s in %s in the %s calendar is -> %d %d %d %d %d %f'%(
           str(off), units, calendar, y, mo, d, h, mn, s))

    if y == 2020 and mo == 8 and d == 10 and h == 19 and mn == 0 and s == 0:
        status('OK!')
    else:
        status('ERROR')

# test passing numpy scalars and Python types
# passing floating point types here would require a cast to an integer type
years = [np.int16(2020), np.int32(2020), np.int64(2020),
         int(2020)]

for year in years:
    status('testing is_leap_year on type %s = %d ... '%(str(type(year)), year))

    leap = calendar_util.is_leap_year(calendar, units, year)

    yn = 'is' if leap else 'is not'
    status('%d %s a leap year'%(year, yn))

    if leap:
        status('OK!')
    else:
        status('ERROR')

