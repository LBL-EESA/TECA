import os, sys, time
import argparse
import numpy
from tcpyPI import pi as tcpi


class teca_potential_intensity(teca_python_algorithm):
    r""" The teca_potential_intensity command line application computes
    potential intensity (PI) for tropical cyclones using the tcpyPI library
    [Gil21]. Potential intensity is the maximum speed limit of a tropical
    cyclone found by modeling the storm as a thermal heat engine. Because there
    are significant correlations between PI and actual storm wind speeds, PI is
    a useful diagnostic for evaluating or predicting tropical cyclone intensity
    climatology and variability. TECA enables massive amounts of data to be
    processed by the tcpyPI code in parallel. In addition to providing scalable
    high performance I/O needed for accessing large amounts of data, TECA
    handles the necessary pre-processing and post processing tasks such as
    conversions of units, conversions of conventional missing values, and the
    application of land-sea masks(LSM). The LSM should be zero over the ocean
    and 1 over the land.

    For more information see:
    D. M. Gilford. Pypi (v1.3): tropical cyclone potential intensity
    calculations in python. Geoscientific Model Development, 14(5):2351–2369,
    2021. URL: https://gmd.copernicus.org/articles/14/2351/2021/,
    doi:10.5194/gmd-14-2351-2021.
    """

    def __init__(self):

        # input variable names
        self.land_mask_variable = None
        self.sea_level_pressure_variable = None
        self.sea_surface_temperature_variable = None
        self.air_temperature_variable = None
        self.mixing_ratio_variable = None
        self.specific_humidity_variable = None

        # adjustable parameters
        self.exchange_cooeficient_ratio = 0.9
        self.ascent_process_proportion = 0
        self.dissipative_heating = 1
        self.gradient_wind_reduction = 0.8
        self.upper_pressure_level = 50
        self.handle_missing_data = 0
        self.bad_units_abort = 1
        self.land_mask_threshold = 0.5

    def set_land_mask_variable(self, var):
        """ set the name of the optional land sea land_mask variable. If specified
        the land_mask will be applied to the input and output data """
        self.land_mask_variable = var

    def set_sea_level_pressure_variable(self, var):
        """ set the name of the sea level pressure variable """
        self.sea_level_pressure_variable = var

    def set_sea_surface_temperature_variable(self, var):
        """ set the name of the sea surface temperature variable """
        self.sea_surface_temperature_variable = var

    def set_air_temperature_variable(self, var):
        """ set the name of the air temperature variable """
        self.air_temperature_variable = var

    def set_mixing_ratio_variable(self, var):
        """ set the name of the mixing ratio variable """
        self.mixing_ratio_variable = var

    def set_specific_humidity_variable(self, var):
        """ set the name of the mixing ratio variable """
        self.specific_humidity_variable = var

    def set_exchange_cooeficient_ratio(self, val):
        """ set the exchange coefficient ratio Ck/Cd """
        self.exchange_cooeficient_ratio = val

    def set_ascent_process_proportion(self, val):
        """
        select reversible adiabatic ascent (0) or pseudoadiabatic
        ascent (1)
        """
        self.ascent_process_proportion = val

    def set_dissipative_heating(self, val):
        """ account for dissapitive heating """
        self.dissipative_heating = val

    def set_gradient_wind_reduction(self, val):
        """ scale gradient wind terms by this amount """
        self.gradient_wind_reduction = val

    def set_upper_pressure_level(self, val):
        """
        set the bound below which the input profile is ignored during
        calculation
        """
        self.upper_pressure_level = val

    def set_handle_missing_data(self, val):
        """
        select missing value handling. calculate above missing values (0),
        skip columns with missing values (1)
        """
        self.handle_missing_data = val

    def set_bad_units_abort(self, val):
        """ When set to 1 the program will abort if bad units are detected """
        self.bad_units_abort = val

    def set_bad_units_abort_on(self):
        """ Cause the program to abort if bad units are detected """
        self.bad_units_abort = 1

    def set_bad_units_abort_off(self):
        """ Cause the program to run even if bad units are detected """
        self.bad_units_abort = 0

    def set_land_mask_threshold(self, val):
        """
        set the threshold above which is cell is considered over land and a
        solution is not computed for
        """
        self.land_mask_threshold = val

    def report(self, port, rep_in):
        """ TECA report override """
        rep_start = time.monotonic_ns()
        rank = self.get_communicator().Get_rank()

        # check on required run time settings
        if rank == 0 and self.sea_level_pressure_variable is None:
            raise RuntimeError('[%d] ERROR: sea_level_pressure_variable was'
                               ' not set.' % (rank))

        if rank == 0 and self.sea_surface_temperature_variable is None:
            raise RuntimeError('[%d] ERROR: sea_surface_temperature_variable'
                               ' was not set.' % (rank))

        if rank == 0 and self.air_temperature_variable is None:
            raise RuntimeError('[%d] ERROR: air_temperature_variable was not'
                              ' set.' % (rank))

        if rank == 0 and self.mixing_ratio_variable is None and \
            self.specific_humidity_variable is None:
            raise RuntimeError('[%d] ERROR: Neither mixing_ratio_variable nor'
                               ' specific_humidity_variable were set.' % (rank))

        # copy the incoming report
        rep = rep_in[0]

        # fix dimensionality. input is 3D output is 2D
        if rep.has('whole_extent'):
            wext = rep['whole_extent']
            wext[4] = 0
            wext[5] = 0
            rep['whole_extent'] = wext

        if rep.has('bounds'):
            bds = rep['bounds']
            bds[4] = 0.0
            bds[5] = 0.0
            rep['bounds'] = bds

        # add the arrays we produce
        out_vars = ['V_max', 'P_min', 'IFL', 'T_o', 'OTL']

        if rep.has('variables'):
            rep.append('variables', out_vars)
        else:
            rep.set('variables', out_vars)

        # get the attributes for a 2D scalar quantity. these are used to
        # boostrap the output attributes for data we produce. This copies the
        # type, size and _FillValue.
        attributes = rep["attributes"]

        psl_atts = attributes[self.sea_level_pressure_variable]

        out_tc = psl_atts['type_code']
        #out_size = psl_atts['size']

        # get a fill value
        if psl_atts.has('_FillValue'):
            out_fill = psl_atts['_FillValue']
        elif psl_atts.has('missing_value'):
            out_fill = psl_atts['missing_value']
        else:
            out_fill = get_default_fill_value(out_tc)

        base_atts = teca_metadata()
        #base_atts['size'] = out_size
        base_atts['type_code'] = out_tc
        base_atts['_FillValue'] = out_fill
        base_atts['mesh_dim_active'] = [1, 1, 0, 1]

        # create attributes for NetCDF CF I/O
        # V_max
        out_atts = teca_metadata(base_atts)

        out_atts['units'] = 'm.s-1'
        out_atts['long_name'] = 'potential intensity'

        attributes['V_max'] = out_atts

        # P_min
        out_atts = teca_metadata(base_atts)

        out_atts['units'] = 'hPa'
        out_atts['long_name'] =  'minimum central pressure'

        attributes['P_min'] = out_atts

        # IFL
        out_atts = teca_metadata()

        ifl_tc = teca_int_array_code.get()

        out_atts['type_code'] = ifl_tc
        out_atts['_FillValue'] = get_default_fill_value(ifl_tc)
        out_atts['units'] = 'unitless'
        out_atts['long_name']= 'algorithm status flag'
        out_atts['mesh_dim_active'] = [1, 1, 0, 1]

        out_atts['description'] = \
            '0 = bad input, 1 = success, 2 = fail to converge, 3 = missng value'

        attributes['IFL'] = out_atts

        # T_o
        out_atts = teca_metadata(base_atts)

        out_atts['units'] = 'K'
        out_atts['long_name'] = 'outflow temperature'

        attributes['T_o'] = out_atts

        # OTL
        out_atts = teca_metadata(base_atts)

        out_atts['units'] = 'hPa'
        out_atts['long_name'] = 'outflow temperature level'

        attributes['OTL'] = out_atts

        # update the attributes collection
        rep["attributes"] = attributes

        if rank == 0 and self.get_verbose() > 1:

            rep_end = time.monotonic_ns()

            sys.stderr.write('[%d] STATUS: teca_potential_intensity::report'
                             ' %f seconds\n' % (rank,
                             (rep_end - rep_start) / 1e9))

        return rep


    def request(self, port, md_in, req_in):
        """ TECA request override """
        req_start = time.monotonic_ns()
        rank = self.get_communicator().Get_rank()


        # cpoy the incoming request to preserve down stream requirements
        req = teca_metadata(req_in)

        # get the list of requested arrays
        arrays = req['arrays']

        # remove the arrays we produce
        out_vars = ['V_max', 'P_min', 'IFL', 'T_o', 'OTL']

        for out_var in out_vars:
            if out_var in arrays:
                arrays.remove(out_var)

        # add the arrays we need
        mr_or_hus_var = self.mixing_ratio_variable
        if mr_or_hus_var is None:
            mr_or_hus_var = self.specific_humidity_variable

        arrays += [ self.sea_level_pressure_variable, \
                    self.sea_surface_temperature_variable, \
                    self.air_temperature_variable, mr_or_hus_var]

        # reaquest valid value land_masks
        vars_in = md_in[0]['variables']
        if self.sea_level_pressure_variable + '_valid' in vars_in:
            arrays += [ self.sea_level_pressure_variable + '_valid', \
                        self.sea_surface_temperature_variable + '_valid', \
                        self.air_temperature_variable + '_valid', \
                        mr_or_hus_var + '_valid']

        # requast the land sea land_mask
        if self.land_mask_variable is not None:
            arrays += [self.land_mask_variable]

        # update the requested arrays
        req['arrays'] = arrays

        # fix dimensionality. input is 3D output is 2D
        md = md_in[0]
        wext = md['whole_extent']

        if md.has('bounds'):
            bds_in = md['bounds']
        else:
            coords = md['coordinates']
            x = coords['x'].as_array()
            y = coords['y'].as_array()
            z = coords['z'].as_array()
            bds_in = [x[0], x[-1], y[0], y[-1], z[0], z[-1]]

        if req.has('bounds'):
            bds = req['bounds']
            bds[4] = bds_in[4]
            bds[5] = bds_in[5]
            req['bounds'] = bds

        if req.has('extent'):
            ext = req['extent']
            ext[4] = wext[4]
            ext[5] = wext[5]
            req['extent'] = ext

        if self.get_verbose() > 1:

            req_end = time.monotonic_ns()

            sys.stderr.write('[%d] STATUS: teca_potential_intensity::request'
                             ' %f seconds\n' % (rank, (req_end - req_start) / 1e9))

        return [req]

    def execute(self, port, data_in, req):
        """ TECA execute override """
        np = numpy

        exec_start = time.monotonic_ns()
        rank = self.get_communicator().Get_rank()
        verbose = self.get_verbose()

        # get the input data
        in_mesh = as_teca_cartesian_mesh(data_in[0])

        # get the mesh extents
        ext = in_mesh.get_extent()

        nx = ext[1] - ext[0] + 1
        ny = ext[3] - ext[2] + 1
        nz = ext[5] - ext[4] + 1
        nxy = nx*ny

        # get the input arrays
        in_arrays = in_mesh.get_point_arrays()

        psl = in_arrays[self.sea_level_pressure_variable].get_host_accessible()
        sst = in_arrays[self.sea_surface_temperature_variable].get_host_accessible()
        ta = in_arrays[self.air_temperature_variable].get_host_accessible()

        if self.mixing_ratio_variable is not None:
            specific_humidity_to_mixing_ratio = False
            mr_var = self.mixing_ratio_variable
        else:
            specific_humidity_to_mixing_ratio = True
            mr_var = self.specific_humidity_variable
        mr = in_arrays[mr_var].get_host_accessible()

        # get the land_mask variable. the land mask is expected to be zero over
        # the ocean and greater than zero over land.
        if self.land_mask_variable is not None:
            land_mask = in_arrays[self.land_mask_variable].get_host_accessible()
            land_mask = np.where(land_mask > self.land_mask_threshold,
                                 np.int8(0), np.int8(1))
        else:
            land_mask = np.ones(nxy, dtype=np.int8)

        # get the pressure coordinate
        plev = in_mesh.get_z_coordinates().as_array()
        plev_var = in_mesh.get_z_coordinate_variable()

        # get the array attributes
        atts = in_mesh.get_attributes()

        # replace _FillValue with NaN
        psl_valid_var = self.sea_level_pressure_variable + '_valid'
        if in_arrays.has(psl_valid_var):
            psl_valid = in_arrays[psl_valid_var].get_host_accessible()
            ii = np.where(np.logical_not(psl_valid))[0]
            psl[ii] = np.NAN

        sst_valid_var = self.sea_surface_temperature_variable + '_valid'
        if in_arrays.has(sst_valid_var):
            sst_valid = in_arrays[sst_valid_var].get_host_accessible()
            ii = np.where(np.logical_not(sst_valid))[0]
            sst[ii] = np.NAN

        ta_valid_var = self.air_temperature_variable + '_valid'
        if in_arrays.has(ta_valid_var):
            ta_valid = in_arrays[ta_valid_var].get_host_accessible()
            ii = np.where(np.logical_not(ta_valid))[0]
            ta[ii] = np.NAN

        mr_valid_var = mr_var + '_valid'
        if in_arrays.has(mr_valid_var):
            mr_valid = in_arrays[mr_valid_var].get_host_accessible()
            ii = np.where(np.logical_not(mr_valid))[0]
            mr[ii] = np.NAN

        # convert pressure coordinate from Pa to hPa
        plev_atts = atts[plev_var]
        plev_units = plev_atts['units']
        if plev_units == 'Pa':

            if rank == 0 and verbose:
                sys.stderr.write('[%d] STATUS: converting pressure coordinate %s'
                                 ' from Pa to hPa\n' % (rank, plev_var))
            plev /= 100.0

        elif plev_units != 'hPa':

            if rank == 0:
                sys.stderr.write('[%d] ERROR: unsupported units %s for pressure'
                                 ' coordinate %s. Pressure must be specified in'
                                 ' either Pa or hPa\n' % (rank, plev_units, plev_var))

            if self.bad_units_abort:
                sys.stderr.flush()
                os.abort()

        # convert sea surface pressure from Pa to hPa
        psl_atts = atts[self.sea_level_pressure_variable]
        psl_units = psl_atts['units']
        if psl_units == 'Pa':

            if rank == 0 and verbose:
                sys.stderr.write('[%d] STATUS: converting seal level pressure %s from'
                                 ' Pa to hPa\n' % (rank, self.sea_level_pressure_variable))
            psl /= 100.0

        elif psl_units != 'hPa':

            if rank == 0:
                sys.stderr.write('[%d] ERROR: unsupported units %s for sea surface'
                                 'pressure %s. Pressure must be specified in'
                                 ' either Pa or hPa\n' % (rank, psl_units,
                                 self.sea_level_pressure_variable))

            if self.bad_units_abort:
                sys.stderr.flush()
                os.abort()

        # convert air temperature from Kelvin to Celcius
        ta_atts = atts[self.air_temperature_variable]
        ta_units = ta_atts['units']
        if ta_units == 'K' or ta_units == 'degrees K':

            if rank == 0 and verbose:
                sys.stderr.write('[%d] STATUS: converting air temperature %s from'
                                 ' degrees K to degrees C\n' % (rank,
                                 self.air_temperature_variable))
            ta -= 273.15

        elif ta_units != 'C' and ta_units != 'degrees C':

            if rank == 0:
                sys.stderr.write('[%d] ERROR: unrecognized units %s for %s.'
                                 ' Temperature must be specified in either C'
                                 ' or K\n' % (rank, ta_units,
                                 self.air_temperature_variable))

            if self.bad_units_abort:
                sys.stderr.flush()
                os.abort()

        # convert sea surface temperature from Kelvin to Celcius
        sst_atts = atts[self.sea_surface_temperature_variable]
        sst_units = sst_atts['units']
        if sst_units == 'K' or sst_units == 'degrees K':

            if rank == 0 and verbose:
                sys.stderr.write('[%d] STATUS: converting sea surface temperature %s from'
                                 ' degrees K to degrees C\n' % (rank,
                                 self.sea_surface_temperature_variable))
            sst -= 273.15

        elif sst_units != 'C' and sst_units != 'degrees C':

            if rank == 0:
                sys.stderr.write('[%d] ERROR: unrecognized units %s for %s.'
                                 ' Temperature must be specified in either C'
                                 ' or K\n' % (rank, sst_units,
                                 self.sea_surface_temperature_variable))

            if self.bad_units_abort:
                sys.stderr.flush()
                os.abort()

        # convert specific humidity to mixing ratio
        if specific_humidity_to_mixing_ratio:

            # check the units
            hus_atts = atts[self.specific_humidity_variable]
            hus_units = hus_atts['units']

            if hus_units == 'kg/kg' or hus_units == 'kg.kg^-1' \
                or hus_units == 'kg kg^-1'or hus_units == 'kg.kg**-1' \
                or hus_units == 'kg kg**-1' or hus_units == '1':

                if rank == 0 and verbose:
                    sys.stderr.write('[%d] STATUS: converting specific humidity %s from'
                                     ' %s to g/kg\n' % (rank,
                                     self.specific_humidity_variable, hus_units))
                mr *= 1.0e3

            elif hus_units != 'g/kg':

                if rank == 0:
                    sys.stderr.write('[%d] ERROR: unrecognized units %s for %s.'
                                     ' Specific humidity must be specified in units'
                                     ' of g/kg\n' % (rank, hus_units,
                                     self.specific_humidity_variable))

                if self.bad_units_abort:
                    sys.stderr.flush()
                    os.abort()

            if rank == 0 and verbose:
                sys.stderr.write('[%d] STATUS: converting specific humidity %s to a'
                                 ' mixing ratio\n' % (rank, self.specific_humidity_variable))

            mr = mr / (1.0 - mr)

        else:

            # check the units
            mr_atts = atts[self.mixing_ratio_variable]
            mr_units = mr_atts['units']

            if mr_units == 'kg/kg' or mr_units == 'kg.kg^-1' \
                or mr_units == 'kg kg^-1'or mr_units == 'kg.kg**-1' \
                or mr_units == 'kg kg**-1' or mr_units == '1':

                if rank == 0 and verbose:
                    sys.stderr.write('[%d] STATUS: converting mixing ratio %s from'
                                     ' %s to g/kg\n' % (rank,
                                     self.mixing_ratio_variable, mr_units))
                mr *= 1.0e3

            elif mr_units != 'g/kg':

                if rank == 0:
                    sys.stderr.write('[%d] ERROR: unrecognized units %s for %s.'
                                     ' Mixing ratio must be specified in units'
                                     ' of g/kg\n' % (rank, mr_units,
                                     self.mixing_ratio_variable))

                if self.bad_units_abort:
                    sys.stderr.flush()
                    os.abort()

        # report min/max for debug
        if verbose > 2:
            sys.stderr.write('%s range [%g, %g]\n' % (
                             self.sea_level_pressure_variable,
                             np.nanmin(psl), np.nanmax(psl)))
            sys.stderr.write('%s range [%g, %g]\n' % (
                             self.sea_surface_temperature_variable,
                             np.nanmin(sst), np.nanmax(sst)))
            sys.stderr.write('%s range [%g, %g]\n' % (
                             self.air_temperature_variable,
                             np.nanmin(ta), np.nanmax(ta)))
            sys.stderr.write('%s range [%g, %g]\n' % (
                             mr_var, np.nanmin(mr), np.nanmax(mr)))

        # allocate output arrays
        vmax = np.empty(nxy, dtype=psl.dtype)
        pmin = np.empty(nxy, dtype=psl.dtype)
        ifl = np.empty(nxy, dtype=np.int32)
        to = np.empty(nxy, dtype=psl.dtype)
        otl = np.empty(nxy, dtype=psl.dtype)

        # for each column compute potential intensity
        j = 0
        while j < ny:

            i = 0
            while i < nx:

                # get the column index
                q = j*nx + i

                # extract columns
                ta_q = ta[q::nxy]
                mr_q = mr[q::nxy]

                # compute potential intensity
                land_mask_q = land_mask[q]
                if land_mask_q:
                    try:

                        vmax[q], pmin[q], ifl[q], to[q], otl[q] = \
                            tcpi ( sst[q], psl[q], plev, ta_q, mr_q,
                                   CKCD = self.exchange_cooeficient_ratio,
                                   diss_flag = self.dissipative_heating,
                                   ptop = self.upper_pressure_level,
                                   miss_handle = self.handle_missing_data )

                    except Exception as details:

                        sys.stderr.write('ERROR: tcPyPI library call failed\n')
                        sys.stderr.write('i = %d, j = %d, q = %d\n' % ( i, j, q ) )
                        sys.stderr.write('sst[q] = %g, psl[q] = %g\n' % ( sst[q], psl[q]) )
                        sys.stderr.write('plev = %s\n' % ( str(plev) ) )
                        sys.stderr.write('ta[q] = %s\n' % ( str(ta_q) ) )
                        sys.stderr.write('mr[q] = %s\n' % ( str(mr_q) ) )

                        raise details

                else:
                    vmax[q], pmin[q], ifl[q], to[q], otl[q] = \
                        np.NAN, np.NAN, 0, np.NAN, np.NAN

                # next lon
                i += 1

            # next lat
            j += 1

        # replace nan with fill value.
        # set a fill value if none was provided.
        if psl_atts.has('_FillValue'):
            fill_value = psl_atts['_FillValue']
        elif psl_atts.has('missing_value'):
            fill_value = psl_atts['missing_value']
        else:
            fill_value = get_default_fill_value(psl_atts['type_code'])

        ii = np.isnan(vmax)
        vmax[ii] = fill_value

        ii = np.isnan(pmin)
        pmin[ii] = fill_value

        ii = np.isnan(to)
        to[ii] = fill_value

        ii = np.isnan(otl)
        otl[ii] = fill_value

        # construct the output mesh
        out_mesh = teca_cartesian_mesh.New()
        out_mesh.shallow_copy(in_mesh)

        # add the array we computed
        arrays = out_mesh.get_point_arrays()
        arrays['V_max'] = vmax
        arrays['P_min'] = pmin
        arrays['IFL'] = ifl
        arrays['T_o'] = to
        arrays['OTL'] = otl

        if verbose:

            exec_end = time.monotonic_ns()

            sys.stderr.write('[%d] STATUS: teca_potential_intensity::execute'
                             ' %f sec\n' % (rank, (exec_end - exec_start) / 1.0e9))

        return out_mesh
