from teca import *
import sys
import numpy as np

set_stack_trace_on_error()

def isequal(a, b, epsilon):
    return np.fabs(a - b) < epsilon


if not len(sys.argv) == 5:
    sys.stderr.write('test_latitude_damper.py [nx] [ny] ' \
        '[hwhm] [out file]\n')
    sys.exit(-1)

nx = int(sys.argv[1])
ny = int(sys.argv[2])
hwhm = float(sys.argv[3])
out_file = sys.argv[4]

nxy = nx*ny

dx = 360./float(nx - 1)
x = []
for i in range(nx):
    x.append(i*dx)

dy = 180./float(ny - 1)
y = []
for i in range(ny):
    y.append(-90. + i*dy)

x = teca_variant_array.New(x)
y = teca_variant_array.New(y)
z = teca_variant_array.New([0.])
t = teca_variant_array.New([1.])

ones_grid = teca_variant_array.New(np.ones(nxy))

wext = [0, nx - 1, 0, ny - 1, 0, 0]

post_fix = "_damped"

mesh = teca_cartesian_mesh.New()
mesh.set_x_coordinates("lon", x)
mesh.set_y_coordinates("lat", y)
mesh.set_z_coordinates("z", z)
mesh.set_whole_extent(wext)
mesh.set_extent(wext)
mesh.set_time(1.0)
mesh.set_time_step(0)
mesh.get_point_arrays().append("ones_grid", ones_grid)

md = teca_metadata()
md["whole_extent"] = wext
md["time_steps"] = [0]
md["variables"] = ["ones_grid"]
md["number_of_time_steps"] = 1
md["index_initializer_key"] = "number_of_time_steps"
md["index_request_key"] = "time_step"

source = teca_dataset_source.New()
source.set_metadata(md)
source.set_dataset(mesh)

damped_comp = teca_latitude_damper.New()
damped_comp.set_input_connection(source.get_output_port())
damped_comp.set_half_width_at_half_max(hwhm)
damped_comp.set_center(0.0)
damped_comp.append_damped_variable("ones_grid")
damped_comp.set_variable_post_fix(post_fix)

damp_o = teca_dataset_capture.New()
damp_o.set_input_connection(damped_comp.get_output_port())

exe = teca_index_executive.New()
exe.set_start_index(0)
exe.set_end_index(0)

wri = teca_cartesian_mesh_writer.New()
wri.set_input_connection(damp_o.get_output_port())
wri.set_executive(exe)
wri.set_file_name(out_file)

wri.update()

ds = damp_o.get_dataset()
mdo = ds.get_metadata()

out_mesh = teca_cartesian_mesh.New()
out_mesh.copy(ds)

damped_array = out_mesh.get_point_arrays().get("ones_grid" + post_fix)

# find lat index where scalar should be half
hwhm_index = -1
for j in range(ny):
    if isequal(y[j], hwhm, 1e-7):
        hwhm_index = j
        break

# validate the search
if hwhm_index < 0 or hwhm_index > ny:
    sys.stderr.write('Failed to find hwhm index')
    sys.exit(-1)

# check that it is half there
test_val = damped_array[hwhm_index * nx]
if not isequal(test_val, 0.5, 1e-7):
    sys.stderr.write('Value %s at index %s is not 0.5' % \
        (test_val, hwhm_index))
    sys.exit(-1)
