from teca import *
import sys
import numpy as np

set_stack_trace_on_error()

if not len(sys.argv) == 8:
    sys.stderr.write('test_component_area_filter.py [nx] [ny] ' \
        '[num labels x] [num labels y] [low thershold] [consecutive labels] [out file]\n')
    sys.exit(-1)

# 64 random integers between 1000 and 10000 for use as non consecutive labels
labels = [9345, 2548, 5704, 5132, 9786, 8329, 3667, 4332, 6232,
    3775, 2593, 7716, 1212, 9638, 9499, 9284, 6736, 7504, 8273, 5808, 7613,
    1405, 8849, 4405, 4777, 2927, 5903, 5294, 7344, 8335, 8186, 3343, 5341,
    7718, 7614, 6608, 1518, 6246, 7647, 4254, 7719, 6879, 1706, 8408, 1489,
    7054, 9304, 7218, 1275, 4784, 3670, 8859, 8877, 5367, 5340, 1521, 5815,
    5717, 6189, 5342, 4709, 6740, 1804, 6772]

nx = int(sys.argv[1])
ny = int(sys.argv[2])
nyl = int(sys.argv[3])
nxl = int(sys.argv[4])
low_threshold_value = float(sys.argv[5])
consecutive_labels = int(sys.argv[6])
out_file = sys.argv[7]

if not consecutive_labels and (nxl*nyl > 64):
    sys.stderr.write("Max 64 non-consecutive labels")
    sys.exit(-1)

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

cc = []
for j in range(ny):
    yl = int((y[j] + float(90.)) / (float(180.) / nyl)) % nyl
    for i in range(nx):
        xl = int(x[i] / (float(360.) / nxl)) % nxl
        lab = yl*nxl + xl
        cc.append(lab if consecutive_labels else labels[lab])

wext = [0, nx - 1, 0, ny - 1, 0, 0]

post_fix = "_area_filtered"

mesh = teca_cartesian_mesh.New()
mesh.set_x_coordinates("lon", x)
mesh.set_y_coordinates("lat", y)
mesh.set_z_coordinates("z", z)
mesh.set_whole_extent(wext)
mesh.set_extent(wext)
mesh.set_time(1.0)
mesh.set_time_step(0)
mesh.get_point_arrays().append("labels", cc)

md = teca_metadata()
md["whole_extent"] = wext
md["variables"] = ["cc"]
md["number_of_time_steps"] = 1
md["index_initializer_key"] = "number_of_time_steps"
md["index_request_key"] = "time_step"

source = teca_dataset_source.New()
source.set_metadata(md)
source.set_dataset(mesh)

ca = teca_2d_component_area.New()
ca.set_input_connection(source.get_output_port())
ca.set_component_variable("labels")
ca.set_contiguous_component_ids(consecutive_labels)

caf = teca_component_area_filter.New()
caf.set_input_connection(ca.get_output_port())
caf.set_component_variable("labels")
caf.set_component_ids_key("component_ids")
caf.set_component_area_key("component_area")
caf.set_low_area_threshold(low_threshold_value)
caf.set_variable_post_fix(post_fix)

cao = teca_dataset_capture.New()
cao.set_input_connection(caf.get_output_port())

exe = teca_index_executive.New()
exe.set_start_index(0)
exe.set_end_index(0)

wri = teca_cartesian_mesh_writer.New()
wri.set_input_connection(cao.get_output_port())
wri.set_executive(exe)
wri.set_file_name(out_file)

wri.update()

va = cao.get_dataset()
mdo = va.get_metadata()

filtered_label_id = []

out_mesh = teca_cartesian_mesh.New()
out_mesh.copy(va)
filtered_labels_all = out_mesh.get_point_arrays().get("labels" + post_fix)

component_ids = mdo["component_ids"]
component_area = mdo["component_area"]

component_ids_filtered = mdo["component_ids" + post_fix]
component_area_filtered = mdo["component_area" + post_fix]

for i in range(len(component_ids)):
    if component_area[i] < low_threshold_value:
        filtered_label_id.append(component_ids[i])

n_filtered = len(filtered_label_id)
n_labels_total = len(filtered_labels_all)
for i in range(n_filtered):
    label = filtered_label_id[i]
    for j in range(n_labels_total):
        if label == filtered_labels_all[j]:
            sys.stderr.write("\nArea filter failed!\n\n")
            sys.exit(-1)
