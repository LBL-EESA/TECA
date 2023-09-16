import numpy
if get_teca_has_cupy():
    import cupy
import sys


class teca_lapse_rate(teca_python_vertical_reduction):

    t_var = "T"
    z_var = "Z"
    zs_var = "ZS"
    zmax = 9000
    z_is_geopotential = True
    fill_value = 1e20
    gravity = 9.806

    def set_t_var(self, t_var):
        """ Sets the temperature variable name. """
        self.t_var = t_var

    def set_z_var(self, z_var):
        """ Sets the geopotential height variable name. """
        self.z_var = z_var

    def set_zs_var(self, zs_var):
        """ Sets the surface geopotential height variable name. """
        self.zs_var = zs_var

    def set_zmax(self, zmax):
        """ Sets the maximum height to use in the lapse rate calculation. """
        self.zmax = zmax

    def set_geopotential_flag(self, geopotential_flag):
        """ Flags whether z is geopotential (units m^2/s^2) or height (m) """
        self.z_is_geopotential = geopotential_flag

    def set_fill_value(self, fill_value):
        self.fill_value = fill_value

    def get_fill_value(self, fill_value):
        return self.fill_value

    def get_point_array_names(self):
        """ Returns the names of the output arrays """
        return ["lapse_rate"]

    def calculate_lapse_rate(self, dev, t, z, zs, zmax=9000):
        """ Calculates the mean lapse rate.

            input:
            ------

                t  : atmospheric temperature [K]

                z  : geopotential height [m]

                zs : surface geopotential height [m]

                zmax : the maximum height to consider in the lapse rate
                       calculation [m]

            output:
            -------

                dtdz : the mean lapse rate [K/m]


        Uses 1st order finite difference to estimate the lapse rate at all
        available levels and then averages. Levels below the surface are
        masked, and levels above zmax are masked.

        It is assumed that the vertical dimension is the first dimension
        e.g., [level, lat, lon]) for both t and z.

        """

        # check shapes
        if t.shape != z.shape:
            raise RuntimeError('ERROR: t and z must have the same shape;'
                    ' shape(t) = %s but shape(z) = %s' % (str(t.shape),str(z.shape)))
        if z[0, ...].shape != zs.shape:
            raise RuntimeError('ERROR: All but the first dimension'
                   ' of z must match the dimensions of zs;'
                   ' shape(z) = %s but shape(zs) = %s' % (str(z.shape),str(zs.shape)))

        # select GPU or CPU
        if dev < 0:
            np = numpy
        else:
            np = cupy

        # mask temperature and height
        # for under-topo values and values that are above the max height
        t_mask = np.where((z >= zs[np.newaxis, ...]) & (z < zmax), t, np.nan)
        z_mask = np.where((z >= zs[np.newaxis, ...]) & (z < zmax), z, np.nan)

        # calculate the lapse rate between each level
        dt = np.diff(t_mask, axis=0)
        dz = np.diff(z_mask, axis=0)

        # calculate the average lapse rate
        dtdz = np.nanmean((dt/dz), axis=0)

        return dtdz

    def request(self, port, md_in, req_in):
        """ Define the TECA request phase. """

        self.set_dependent_variables(
            [
                self.t_var,
                self.z_var,
                self.zs_var,
            ]
        )

        return super().request(port, md_in, req_in)

    def report(self, port, md_in):
        """ Define the TECA report phase """
        md = teca_metadata(md_in[0])

        lapse_atts = teca_array_attributes(
            teca_double_array_code.get(), teca_array_attributes.point_centering,
            0, teca_array_attributes.xyt_active(), 'k/m', 'Average Lapse Rate',
            "Average lapse rate below {self.zmax} m", self.fill_value)

        # add the variables
        self.add_derived_variable_and_attributes("lapse_rate", lapse_atts)

        return super().report(port, md_in)

    def execute(self, port, data_in, req):
        """Define the TECA execute phase for this algorithm.
           Outputs a 2D array of lapserates. """

        # get the device to run on
        dev = -1
        np = numpy
        if get_teca_has_cuda() and get_teca_has_cupy():
            dev = req['device_id']
            if dev >= 0:
                cupy.cuda.Device(dev).use()
                np = cupy
        # report
        if self.get_verbose():
            dev_str = 'CPU' if dev < 0 else 'GPU %d' % (dev)
            sys.stderr.write('teca_lapse_rate::execute %s\n' % (dev_str))

        # get the input mesh
        in_mesh = as_const_teca_cartesian_mesh(data_in[0])

        # initialize the output mesh from the super function
        out_mesh = as_const_teca_cartesian_mesh(
                   super().execute(port, data_in, req))

        # get the pressure level
        level = in_mesh.get_z_coordinates()

        # get horizontal coordinates
        lon = in_mesh.get_x_coordinates()
        lat = in_mesh.get_y_coordinates()

        def reshape3d(in_var):
            return np.reshape(in_var, [len(level), len(lat), len(lon)])

        def reshape2d(in_var):
            return np.reshape(in_var, [len(lat), len(lon)])

        arrays = in_mesh.get_point_arrays()
        if dev < 0:
            in_t = arrays[self.t_var].get_host_accessible()
            in_z = arrays[self.z_var].get_host_accessible()
            in_zs = arrays[self.zs_var].get_host_accessible()
        else:
            in_t = arrays[self.t_var].get_cuda_accessible()
            in_z = arrays[self.z_var].get_cuda_accessible()
            in_zs = arrays[self.zs_var].get_cuda_accessible()

        # get temperature, surface geopotential and level geopotential
        t = reshape3d(in_t)
        z = reshape3d(in_z)
        zs = reshape2d(in_zs)

        # convert to units of meters if needd
        if self.z_is_geopotential:
            z /= self.gravity
            zs /= self.gravity

        # calculate cloud base height
        lapse_rate = self.calculate_lapse_rate(
            dev,
            t.astype(np.float64),
            z.astype(np.float64),
            zs.astype(np.float64),
            zmax=self.zmax
        )

        out_arrays = out_mesh.get_point_arrays()
        out_arrays['lapse_rate'] = lapse_rate.ravel()

        # return the current lapse rate mesh
        return out_mesh
