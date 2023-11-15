from teca import *
import sys

data_root = sys.argv[1]
infn ='%s/e5\.oper\.an\.pl\.128_129_z\.ll025sc\.2020010100_2020010123\.nc$'%(data_root)
var = 'Z'
x_var = 'longitude'
y_var = 'latitude'
z_var = 'level'

cfr = teca_cf_reader.New()
cfr.set_files_regex(infn)
cfr.set_x_axis_variable(x_var)
cfr.set_y_axis_variable(y_var)
cfr.set_z_axis_variable(z_var)

# require that the input is not already in units of Pa
# this is true for ERA5 datasets
sys.stderr.write('\n\nchecking original z-axis units ... ')

md = cfr.update_metadata()

atts = md.get("attributes")
z_atts = atts.get(z_var)
z_units = z_atts.get("units")

if z_units == "Pa":
    sys.stderr.write('\n\nERROR: z_units == %s \n\n'%(z_units))
    sys.exit(-1)

sys.stderr.write('%s OK!\n'%(z_units))

# display the input coordinates
coords = md.get("coordinates")
z = coords.get("z")
sys.stderr.write('z_in = %s\n'%(str(z)))

# display the input bounds
bounds = md.get("bounds")
sys.stderr.write('bounds_in = %s\n'%(str(bounds)))


nc = teca_normalize_coordinates.New()
nc.set_enable_unit_conversions(1)
nc.set_input_connection(cfr.get_output_port())
nc.set_verbose(0)


# check metadata
sys.stderr.write('\n\nchecking metadata for z-axis units of Pa ... ')

md = nc.update_metadata()
atts = md.get("attributes")
z_atts = atts.get(z_var)
z_units = z_atts.get("units")

if z_units != "Pa":
    sys.stderr.write('\n\nERROR: z_units == %s != Pa\n\n'%(z_units))
    sys.exit(-1)

sys.stderr.write('OK!\n')

# display the transformed coordinates
coords = md.get("coordinates")
z = coords.get("z")
sys.stderr.write('report z = %s\n'%(str(z)))

# display the input bounds
bounds = md.get("bounds")
sys.stderr.write('report bounds = %s\n'%(str(bounds)))




dsc = teca_dataset_capture.New()
dsc.set_input_connection(nc.get_output_port())

exe = teca_index_executive.New()
exe.set_start_index(0)
exe.set_end_index(0)
exe.set_arrays([var])
exe.set_bounds(bounds)

dsc.set_executive(exe)


# check mesh
sys.stderr.write('\n\nchecking mesh for z-axis units of Pa...')

dsc.update()

mesh = as_teca_cartesian_mesh(dsc.get_dataset())
atts = mesh.get_attributes()
z_atts = atts.get(z_var)
z_units = z_atts.get("units")

if z_units != "Pa":
    sys.stderr.write('\n\nERROR: z_units == %s != Pa\n\n'%(z_units))
    sys.exit(-1)

sys.stderr.write('OK!\n')

z = mesh.get_z_coordinates()
sys.stderr.write('mesh z = %s\n'%(str(z)))

bounds = mesh.get_bounds()
sys.stderr.write('mesh bounds = %s\n'%(str(bounds)))

sys.exit(0)
