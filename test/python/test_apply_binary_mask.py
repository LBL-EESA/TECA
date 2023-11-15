from teca import *
import sys
import numpy as np

set_stack_trace_on_error()

def isequal(a, b, epsilon):
    return np.fabs(a - b) < epsilon


if not len(sys.argv) == 5:
    sys.stderr.write('test_apply_binary_mask.py [nx] [ny] [prefix] [tolerance]\n')
    sys.exit(-1)

nx = int(sys.argv[1])
ny = int(sys.argv[2])
prefix = str(sys.argv[3])
tolerance = float(sys.argv[4])

nxy = nx*ny

dx = 360./float(nx - 1)
x = []
for i in range(nx):
    x.append(i*dx)

dy = 180./float(ny - 1)
y = []
for i in range(ny):
    y.append(-90. + i*dy)

# create a mask variable representing cosine-latitude weighting
coslat = np.cos(np.deg2rad(np.array(y)[:,np.newaxis]))*np.ones([ny, nx])
# normalize it so it ranges from 0 to 1
coslat /= np.sum(coslat)


x = teca_variant_array.New(x)
y = teca_variant_array.New(y)
z = teca_variant_array.New([0.])
t = teca_variant_array.New([1.])
mask_grid = teca_variant_array.New(coslat.astype(np.float32).ravel())

ones_grid = teca_variant_array.New(np.ones(nxy).astype(np.float64))
zeros_grid = teca_variant_array.New(np.zeros(nxy).astype(np.float64))
mask_grid = teca_variant_array.New(mask_grid)

wext = [0, nx - 1, 0, ny - 1, 0, 0]

mesh = teca_cartesian_mesh.New()
mesh.set_x_coordinates("lon", x)
mesh.set_y_coordinates("lat", y)
mesh.set_z_coordinates("z", z)
mesh.set_whole_extent(wext)
mesh.set_extent(wext)
mesh.set_time(1.0)
mesh.set_time_step(0)
mesh.get_point_arrays().append("ones_grid", ones_grid)
mesh.get_point_arrays().append("zeros_grid", zeros_grid)
mesh.get_point_arrays().append("mask_grid", mask_grid)

md = teca_metadata()
md["whole_extent"] = wext
md["time_steps"] = [0]
md["variables"] = ["ones_grid", "zeros_grid", "mask_grid"]
md["number_of_time_steps"] = 1
md["index_initializer_key"] = "number_of_time_steps"
md["index_request_key"] = "time_step"

# add attributes
ones_atts = teca_array_attributes(
    teca_double_array_code.get(),
    teca_array_attributes.no_centering,
    2, (0,0,0,0), 'ones', 'unitless',
    'an array full of ones',
    None)

zeros_atts = teca_array_attributes(
    teca_double_array_code.get(),
    teca_array_attributes.no_centering,
    2, (0,0,0,0), 'zeros', 'unitless',
    'an array full of zeros',
    None)

# put it in the array attributes
try:
    atts = md['attributes']
except:
    atts = teca_metadata()
atts['ones_grid'] = ones_atts.to_metadata()
atts['zeros_grid'] = zeros_atts.to_metadata()
md['attributes'] = atts



source = teca_dataset_source.New()
source.set_metadata(md)
source.set_dataset(mesh)

mask_comp = teca_apply_binary_mask.New()
mask_comp.set_input_connection(source.get_output_port())
mask_comp.set_mask_variable("mask_grid")
mask_comp.set_masked_variables(["ones_grid", "zeros_grid"])
mask_comp.set_output_variable_prefix(prefix)

mask_o = teca_dataset_capture.New()
mask_o.set_input_connection(mask_comp.get_output_port())

mask_o.update()

ds = mask_o.get_dataset()
mdo = ds.get_metadata()

out_mesh = teca_cartesian_mesh.New()
out_mesh.copy(ds)

out_arrays = out_mesh.get_point_arrays()

masked_ones_array = out_arrays[prefix + "ones_grid"].get_host_accessible()
masked_zeros_array = out_arrays[prefix + "zeros_grid"].get_host_accessible()

# check that the sum of ones times the mask is equal to 1
sum_difference = np.sum(masked_ones_array) - 1
if np.abs(sum_difference) > tolerance:
    sys.stderr.write('Failure: sum of dx*dy field ' +
        'differs from 1 by {:2.2g}\n'.format(sum_difference))
    sys.exit(-1)

# check that the sum of zeros is zero
sum_difference = np.sum(masked_zeros_array)
if not np.isclose(sum_difference, 0):
    sys.stderr.write('Failure: sum of zeros field ' +
        'differs from 0 by {:2.2g}\n'.format(sum_difference))
    sys.exit(-1)
