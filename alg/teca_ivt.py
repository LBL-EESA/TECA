import sys
import teca_py
import numpy as np

# first attempt to import simps from scipy.integrate
try:
    from scipy.integrate import simps
# if that doesn't work, define
# the simps() function based on scipy.integrate version 0.19.1
# TODO: remove this, since it is likely affects the license status of TECA
except:
    def tupleset(t, i, value):
        l = list(t)
        l[i] = value
        return tuple(l)

    def _basic_simps(y, start, stop, x, dx, axis):
        nd = len(y.shape)
        if start is None:
            start = 0
        step = 2
        slice_all = (slice(None),)*nd
        slice0 = tupleset(slice_all, axis, slice(start, stop, step))
        slice1 = tupleset(slice_all, axis, slice(start+1, stop+1, step))
        slice2 = tupleset(slice_all, axis, slice(start+2, stop+2, step))

        if x is None:  # Even spaced Simpson's rule.
            result = np.sum(dx/3.0 * (y[slice0]+4*y[slice1]+y[slice2]),
                            axis=axis)
        else:
            # Account for possibly different spacings.
            #    Simpson's rule changes a bit.
            h = np.diff(x, axis=axis)
            sl0 = tupleset(slice_all, axis, slice(start, stop, step))
            sl1 = tupleset(slice_all, axis, slice(start+1, stop+1, step))
            h0 = h[sl0]
            h1 = h[sl1]
            hsum = h0 + h1
            hprod = h0 * h1
            h0divh1 = h0 / h1
            tmp = hsum/6.0 * (y[slice0]*(2-1.0/h0divh1) +
                              y[slice1]*hsum*hsum/hprod +
                              y[slice2]*(2-h0divh1))
            result = np.sum(tmp, axis=axis)
        return result

    def simps(y, x=None, dx=1, axis=-1, even='avg'):
        """
        Integrate y(x) using samples along the given axis and the composite
        Simpson's rule.  If x is None, spacing of dx is assumed.
        If there are an even number of samples, N, then there are an odd
        number of intervals (N-1), but Simpson's rule requires an even number
        of intervals.  The parameter 'even' controls how this is handled.
        Parameters
        ----------
        y : array_like
            Array to be integrated.
        x : array_like, optional
            If given, the points at which `y` is sampled.
        dx : int, optional
            Spacing of integration points along axis of `y`. Only used when
            `x` is None. Default is 1.
        axis : int, optional
            Axis along which to integrate. Default is the last axis.
        even : str {'avg', 'first', 'last'}, optional
            'avg' : Average two results:1) use the first N-2 intervals with
                      a trapezoidal rule on the last interval and 2) use the last
                      N-2 intervals with a trapezoidal rule on the first interval.
            'first' : Use Simpson's rule for the first N-2 intervals with
                    a trapezoidal rule on the last interval.
            'last' : Use Simpson's rule for the last N-2 intervals with a
                   trapezoidal rule on the first interval.
        See Also
        --------
        quad: adaptive quadrature using QUADPACK
        romberg: adaptive Romberg quadrature
        quadrature: adaptive Gaussian quadrature
        fixed_quad: fixed-order Gaussian quadrature
        dblquad: double integrals
        tplquad: triple integrals
        romb: integrators for sampled data
        cumtrapz: cumulative integration for sampled data
        ode: ODE integrators
        odeint: ODE integrators
        Notes
        -----
        For an odd number of samples that are equally spaced the result is
        exact if the function is a polynomial of order 3 or less.  If
        the samples are not equally spaced, then the result is exact only
        if the function is a polynomial of order 2 or less.
        """
        y = np.asarray(y)
        nd = len(y.shape)
        N = y.shape[axis]
        last_dx = dx
        first_dx = dx
        returnshape = 0
        if x is not None:
            x = np.asarray(x)
            if len(x.shape) == 1:
                shapex = [1] * nd
                shapex[axis] = x.shape[0]
                saveshape = x.shape
                returnshape = 1
                x = x.reshape(tuple(shapex))
            elif len(x.shape) != len(y.shape):
                raise ValueError("If given, shape of x must be 1-d or the "
                                 "same as y.")
            if x.shape[axis] != N:
                raise ValueError("If given, length of x along axis must be the "
                                 "same as y.")
        if N % 2 == 0:
            val = 0.0
            result = 0.0
            slice1 = (slice(None),)*nd
            slice2 = (slice(None),)*nd
            if even not in ['avg', 'last', 'first']:
                raise ValueError("Parameter 'even' must be "
                                 "'avg', 'last', or 'first'.")
            # Compute using Simpson's rule on first intervals
            if even in ['avg', 'first']:
                slice1 = tupleset(slice1, axis, -1)
                slice2 = tupleset(slice2, axis, -2)
                if x is not None:
                    last_dx = x[slice1] - x[slice2]
                val += 0.5*last_dx*(y[slice1]+y[slice2])
                result = _basic_simps(y, 0, N-3, x, dx, axis)
            # Compute using Simpson's rule on last set of intervals
            if even in ['avg', 'last']:
                slice1 = tupleset(slice1, axis, 0)
                slice2 = tupleset(slice2, axis, 1)
                if x is not None:
                    first_dx = x[tuple(slice2)] - x[tuple(slice1)]
                val += 0.5*first_dx*(y[slice2]+y[slice1])
                result += _basic_simps(y, 1, N-2, x, dx, axis)
            if even == 'avg':
                val /= 2.0
                result /= 2.0
            result = result + val
        else:
            result = _basic_simps(y, 0, N-2, x, dx, axis)
        if returnshape:
            x = x.reshape(saveshape)
        return result


class teca_ivt(object):
    """ """
    @staticmethod
    def New():
        return teca_ivt()

    @staticmethod
    def calculate_pressure_integral(integrand, pressure):
        """ Calculates the vertical integral (in pressure coordinates) of an array
        
            input:
            ------
                integrand     : the quantity to integrate.  The vertical dimension is assumed to be the first index.
                
                pressure      : the pressure (either a vector or an array of the same shape as integrand).  Units should be [Pa].
                
            output:
            -------
            
                integral      : the approximated integral of integrand (same shape as integrand, but missing the leftmost
                                dimension of integrand).
                                
                For integrand $F(p)$, this function approximates
                
                $$ -\frac{1}{g} \int\limits^{p_s}_0 F(p) dp $$
                
        """
        # set necessary constants
        one_over_negative_g = -1./9.80665 # m/s^2
        
        # determine whether pressure needs to be broadcast to the same shape as integrand
        # check if the dimensions of integrand and pressure don't match
        if not all( [s1 == s2 for s1,s2 in zip(integrand.shape,pressure.shape)] ):
            
            # try broadcasting pressure to the proper shape
            try:
                pressure3d = np.ones(integrand.shape)*pressure[:,np.newaxis,np.newaxis]
            except:
                raise ValueError("pressure cannot be broadcast to the shape of integrand. shape(pressure) = {}, and shape(integrand) = {}".format(pressure.shape,integrand.shape))
        
        # if they do, then simply set pressure3d to pressure
        else:
            pressure3d = pressure
            
        
            # calculate the integral
            # ( fill in any missing values with 0)
            integral = simps(np.ma.filled(integrand,0),pressure,axis=0)
        
        # scale the integral and return
        return one_over_negative_g*integral



    def __init__(self):

        self.arrays = []
        self.bounds = []
        self.do_use_z_coord_as_pressure = False

        self.impl = teca_py.teca_programmable_algorithm.New()
        self.impl.set_number_of_input_connections(1)
        self.impl.set_number_of_output_ports(1)
        self.impl.set_report_callback(self.get_report_callback())
        self.impl.set_request_callback(self.get_request_callback())
        self.impl.set_execute_callback(self.get_execute_callback())

    def set_arrays(self, arrays):
        self.arrays = arrays

    def use_z_coord_as_pressure(self,do_use = True):
        self.do_use_z_coord_as_pressure = do_use

    def set_array_dict(self,**kwargs):
        """ Sets a dictionary of variable names for use in calculating IVT
        
            input:
            ------
                zonal_wind : the variable name corresponding to zonal wind
                
                meridional_wind : the variable name corresponding to meridional wind
                
                water_vapor : the variable name corresponding to water vapor mixing ratio

                pressure : the variable name corresponding to pressure
        
        """

        zonal_wind = kwargs['zonal_wind']
        meridional_wind = kwargs['meridional_wind']
        water_vapor = kwargs['water_vapor']
        if 'pressure' in kwargs:
            pressure = kwargs['pressure']

        self.arrays = [zonal_wind, meridional_wind, water_vapor]

        self.varname_dict = { 'zonal_wind' : zonal_wind, \
                              'meridional_wind' : meridional_wind, \
                              'water_vapor' : water_vapor}
        if 'pressure' in kwargs:
            self.varname_dict['pressure'] = pressure 
            self.arrays.append(pressure)

    def set_bounds(self, bounds):
        if len(bounds) != 6:
            sys.stderr.write('ERROR: bounds is expectecd to have 6 elements!\n')
            return
        self.bounds = bounds

    def set_input_connection(self, port):
        self.impl.set_input_connection(port)

    def get_output_port(self):
        return self.impl.get_output_port()

    def update(self):
        self.impl.update()

    def get_report_callback(self):
        def report_callback(port, md_in):
            # pass through by default
            md_out = teca_py.teca_metadata(md_in[0])
            md_out.append('variables','IVT')
            return md_out
        return report_callback

    def get_request_callback(self):
        def request_callback(port, md_in, req_in):

            # make a copy of the incoming request
            # this preserves down stream needs
            req_out = teca_py.teca_metadata(req_in)

            # get the coordinates
            coords = md_in[0]['coordinates']
            lon = coords['x']
            lat = coords['y']
            lev = coords['z']

            # clamp the user supplied bounds to what's actually
            # available
            req_out.bounds = [max(lon[0], self.bounds[0]),
                min(lon[-1], self.bounds[1]), max(lat[0], self.bounds[2]),
                min(lat[-1], self.bounds[3]), max(lev[0], self.bounds[4]),
                min(lev[-1], self.bounds[5])]

            # add the arrrays we need
            #sys.stderr.write('self.arrays = {}\nreq_out["arrays"] = {}'.format(self.arrays,req_out['arrays']))
            req_out['arrays'] = self.arrays #+ [req_out['arrays']] if \ req_out.has('arrays') else self.arrays

            sys.stderr.write('\n=================teca-ivt request_callback\n%s\n'%(str(req_out)))
            reqs_out = [req_out]
            return reqs_out
        return request_callback

    def get_execute_callback(self):
        def execute_callback(port, data_in, req_in):
            # get the patch
            mesh = teca_py.as_teca_cartesian_mesh(data_in[0])

	    # maybe no mesh if there are more ranks than time steps
            if mesh is None:
                return teca_table.New()

            # coordinates
            lon = mesh.get_x_coordinates().as_array()
            lat = mesh.get_y_coordinates().as_array()
	    lev = mesh.get_z_coordinates().as_array()

            time = mesh.get_time()
            step = mesh.get_time_step()

            nx = len(lon)
            ny = len(lat)
	    nz = len(lev)

            # initialize the array of required variables
            array_dict = {}

            for var in self.varname_dict:
                # get the named scalar
                array = mesh.get_point_arrays().get(self.varname_dict[var]).as_array()

                # convert from a flat 1D array into a 2D array
                array = np.reshape(array, [nz,ny,nx])

                # store the array
                array_dict[var] = array
		# do something here...
                # TODO: remove this
                #sys.stderr.write('got array %s = %s\n'%(var, str(array)))

            if self.do_use_z_coord_as_pressure:
                array_dict['pressure'] = lev

            # calculate IVT vector components
            ivt_u = self.calculate_pressure_integral(array_dict['zonal_wind']*array_dict['water_vapor'], \
                                                     array_dict['pressure'])
            ivt_v = self.calculate_pressure_integral(array_dict['meridional_wind']*array_dict['water_vapor'], \
                                                     array_dict['pressure'])

            # get the magnitude of the IVT vector
            ivt = np.sqrt(ivt_u**2 + ivt_v**2)

            # do something here ...
            return ivt

        return execute_callback
