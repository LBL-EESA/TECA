from teca import *
import sys
import numpy as np

set_stack_trace_on_error()

def isequal(a, b, epsilon):
    return np.fabs(a - b) < epsilon


if not len(sys.argv) == 5:
    sys.stderr.write('test_connected_components.py [nx] [ny] [threshold]\n' \
            '[out_file]')
    sys.exit(-1)

nx = int(sys.argv[1])
ny = int(sys.argv[2])
threshold = float(sys.argv[3])
out_file = str(sys.argv[4])


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

# set the segmentation array
segmentation_array = np.zeros([nx,ny])
# insert a single above-threshold value in the center
segmentation_array[int(nx/2), int(ny/2)] = threshold + 1

test_grid = teca_variant_array.New(segmentation_array.ravel())

wext = [0, nx - 1, 0, ny - 1, 0, 0]

mesh = teca_cartesian_mesh.New()
mesh.set_x_coordinates("lon", x)
mesh.set_y_coordinates("lat", y)
mesh.set_z_coordinates("z", z)
mesh.set_whole_extent(wext)
mesh.set_extent(wext)
mesh.set_time(1.0)
mesh.set_time_step(0)
mesh.get_point_arrays().append("test_grid", test_grid)

md = teca_metadata()
md["whole_extent"] = wext
md["time_steps"] = [0]
md["variables"] = ["test_grid"]
md["number_of_time_steps"] = 1
md["index_initializer_key"] = "number_of_time_steps"
md["index_request_key"] = "time_step"

source = teca_dataset_source.New()
source.set_metadata(md)
source.set_dataset(mesh)

seg = teca_binary_segmentation.New()
seg.set_threshold_variable("test_grid")
seg.set_segmentation_variable("segmented_grid")
seg.set_low_threshold_value(threshold)
seg.set_input_connection(source.get_output_port())

cc = teca_connected_components.New()
cc.set_segmentation_variable("segmented_grid")
cc.set_component_variable("connected_component")
cc.set_input_connection(seg.get_output_port())

seg_o = teca_dataset_capture.New()
seg_o.set_input_connection(cc.get_output_port())

exe = teca_index_executive.New()
exe.set_start_index(0)
exe.set_end_index(0)

wri = teca_cartesian_mesh_writer.New()
wri.set_input_connection(seg_o.get_output_port())
wri.set_executive(exe)
wri.set_file_name(out_file)

wri.update()

# get the dataset
ds = seg_o.get_dataset()
# get the metadata
mdo = ds.get_metadata()

# get the counts
num_components = mdo['number_of_components']

# check that there is only 1
if num_components != 1:
    sys.stderr.write("Number of components is not 1; it is: " 
                     + str(num_components) + "\n")
    sys.exit(-1)
