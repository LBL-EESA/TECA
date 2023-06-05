from teca import *
import math
import numpy
if get_teca_has_cupy():
    import cupy
import sys


class generate_data(teca_python_algorithm):
    """
    This class generates 3D point centered data according to the function:

        tt(x,y,z) = t0 + ((-g/cp) * zz) [adiabatic temperature profile in Kelvin]

        where -g/cp is adiabatic lapse rate and

                    t0 = 288.15
                    g  = 9.81
                    cp = 1004
                    zz(x,y,z) = 0, ..., zmax [height in meters]
                    zmax = 9000 [maximum height in meters]

    and 2D data according to the function:

        zs(x,y) = 0 [surface height in meters]
    """

    def get_point_array_names(self):
        return ['tt', 'zz', 'zs']

    def report(self, port, md_in):

        # report arrays we generate
        md_out = teca_metadata(md_in[0])
        try:
            arrays = md_out['arrays']
        except:
            arrays = []
        md_out['arrays'] = arrays + ['tt', 'zz', 'zs']

        # get the extent of the dataset
        wext = md_out['whole_extent']
        ncells = (wext[1] - wext[0] + 1) * \
                 (wext[3] - wext[2] + 1) * (wext[5] - wext[4] + 1)

        nt = wext[7] - wext[6] + 1
        md_out['bounds'] = [0., 360., -90., 90., 0., nt, 0., 9000.]

        # create the metadata for the writer
        faa = teca_array_attributes(
            teca_double_array_code.get(),
            teca_array_attributes.point_centering,
            int(ncells), (1,1,1,1), 'Kelvin', 'temperature [tt(x,y,z)]',
            't0 + ((-g / cp) * zz)',
            None)

        fatts = faa.to_metadata()

        haa = teca_array_attributes(
            teca_double_array_code.get(),
            teca_array_attributes.point_centering,
            int(ncells), (1,1,1,1), 'meters', 'height [zz(x,y,z)]',
            'none',
            None)

        hatts = haa.to_metadata()

        gaa = teca_array_attributes(
            teca_double_array_code.get(),
            teca_array_attributes.point_centering,
            int(ncells), (1,1,1,1), 'meters', 'surface height [zs(x,y)]',
            'none',
            None)

        gatts = gaa.to_metadata()

        # put it in the array attributes
        try:
            atts = md_out['attributes']
        except:
            atts = teca_metadata()

        atts['tt'] = fatts
        atts['zz'] = hatts
        atts['zs'] = gatts

        md_out['attributes'] = atts
        return md_out

    def execute(self, port, data_in, req_in):
        # get the device to run on
        dev = -1
        np = numpy
        if get_teca_has_cuda() and get_teca_has_cupy():
            dev = req_in['device_id']
            if dev >= 0:
                cupy.cuda.Device(dev).use()
                np = cupy
        # report
        dev_str = 'CPU' if dev < 0 else 'GPU %d' % (dev)
        sys.stderr.write('generate_data::execute %s\n' % (dev_str))

        mesh_in = as_const_teca_cartesian_mesh(data_in[0])

        # get mesh dims and coordinate arrays
        ext = mesh_in.get_extent()
        nx = ext[3] - ext[2] + 1
        ny = ext[1] - ext[0] + 1
        nz = ext[5] - ext[4] + 1

        x = mesh_in.get_x_coordinates().as_array()
        y = mesh_in.get_y_coordinates().as_array()
        z = mesh_in.get_z_coordinates().as_array()
        t = mesh_in.get_time()

        # generate the 3D variable. zz = 0, ..., zmax, zmax = 9000
        zz = np.empty((nz, ny, nx), dtype=np.float64)
        k = 0
        while k < nz:
            zz[k, :, :] = k * (9000. / (nz - 1))
            k += 1

        # generate the 3D variable. tt = t0 + (adiabatic_lapse_rate * zz)
        tt = np.empty((nz, ny, nx), dtype=np.float64)
        k = 0
        while k < nz:
            tt[k, :, :] = 288.15 - ((9.81 / 1004.) * zz[k])
            k += 1

        # generate the 2D variable.
        zs = np.full((ny, nx), 0., dtype=np.float64)

        # create the output and add in the arrays
        mesh_out = teca_cartesian_mesh.New()
        mesh_out.shallow_copy(mesh_in)
        mesh_out.get_point_arrays().append('tt', teca_variant_array.New(tt))
        mesh_out.get_point_arrays().append('zz', teca_variant_array.New(zz))
        mesh_out.get_point_arrays().append('zs', teca_variant_array.New(zs))

        return mesh_out


# process the command line
if not len(sys.argv) == 6:
    sys.stderr.write('usage: test_lapse_rate.py [nx] [ny] [nz] [nt] \
                             [out file]\n')
    sys.exit(-1)

nx = int(sys.argv[1])
ny = int(sys.argv[2])
nz = int(sys.argv[3])
nt = int(sys.argv[4])
out_file = sys.argv[5]

# construct a small mesh
src = teca_cartesian_mesh_source.New()
src.set_x_axis_variable('x')
src.set_y_axis_variable('y')
src.set_z_axis_variable('z')
src.set_t_axis_variable('t')
src.set_whole_extents([0, nx - 1, 0, ny - 1, 0, nz - 1, 0, nt - 1])
src.set_bounds([0., 360., -90., 90., 0., nt, 0., 9000.])
src.set_calendar('standard', 'days since 2022-01-18')

# generate 2d and 3d data
gd = generate_data.New()
gd.set_input_connection(src.get_output_port())

lapse = teca_lapse_rate.New()
lapse.set_input_connection(gd.get_output_port())
lapse.set_t_var("tt")
lapse.set_z_var("zz")
lapse.set_zs_var("zs")
lapse.set_geopotential_flag(False)

lapse_o = teca_dataset_capture.New()
lapse_o.set_input_connection(lapse.get_output_port())

exe = teca_index_executive.New()
exe.set_arrays(["lapse_rate"])
exe.set_bounds([0., 360., -90., 90., 0., 0., 0., 0.])

wri = teca_cartesian_mesh_writer.New()
wri.set_input_connection(lapse_o.get_output_port())
wri.set_executive(exe)
wri.set_file_name(out_file)
# wri.set_binary(0)
# wri.set_output_format(1)

wri.update()

ds = lapse_o.get_dataset()
mdo = ds.get_metadata()

out_mesh = teca_cartesian_mesh.New()
out_mesh.copy(ds)

lapse_array = out_mesh.get_point_arrays().get("lapse_rate")

if not math.isclose(math.fabs(lapse_array[0]*1000), 9.77091633):
    sys.stderr.write('Value %s is not 9.77091633' % \
                     (math.fabs(lapse_array[0]*1000)))
    sys.exit(-1)

if not math.isclose(math.fabs(lapse_array[nx*ny-1]*1000), 9.77091633):
    sys.stderr.write('Value %s is not 9.77091633' % \
                     (math.fabs(lapse_array[nx*ny-1]*1000)))
    sys.exit(-1)
