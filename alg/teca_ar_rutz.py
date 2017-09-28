import numpy as np
import teca_py
import sys

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
    integral = scipy.integrate.simps(np.ma.filled(integrand,0),pressure,axis=0)
    
    # scale the integral and return
    return one_over_negative_g*integral

class teca_ar_rutz(object):
    """ """
    @staticmethod
    def New():
        return teca_ar_rutz()

    def __init__(self):
        self.impl = teca_py.teca_programmable_algorithm.New()
        self.impl.set_number_of_input_connections(1)
        self.impl.set_number_of_output_ports(1)
        self.impl.set_report_callback(self.get_report_callback())
        self.impl.set_request_callback(self.get_request_callback())
        self.impl.set_execute_callback(self.get_execute_callback())

    def set_input_connection(self, port, obj):
        self.impl.set_input_connection(obj)

    def get_output_port(self):
        return self.impl.get_output_port()

    def update(self):
        self.impl.update()

    def get_report_callback(self):
        """ Returns a proper TECA report callback function """
        def report_callback(port, md_in):
            """ Defines the variables that this class returns. """
            # add the names of the variables this method generates
            md_in[0].append('variables', 'IVT')

            return md_in
        
        return report_callback

    def get_request_callback(self):
        """ Returns a proper TECA request callback function """
        def request_callback(port, md_in, req_in):
            """ Requests the variables needed to find ARs """
            # add the name of arrays that we need to find ARs
            req_in['arrays'] = ['QV','u','v','pressure','lat','lon']

            reqs_out = [req_in]
            return reqs_out
        return request_callback

    def get_execute_callback(self):
        """ Returns a proper TECA execute callback function """
        def execute_callback(port, data_in, req_in):

            # pass the incoming data through
            in_mesh = teca_py.as_teca_cartesian_mesh(data_in[0])
            out_mesh = teca_cartesian_mesh.New()
            out_mesh.shallow_copy(in_mesh)

            # extract input arrays
            arrays = out_mesh.get_point_arrays()
            qv = arrays['qv']
            u = arrays['u']
            v = arrays['v']
            # TODO: verify that units of pressure are in hPa
            pressure = arrays['pressure']
            lat = arrays['lat']
            lon = arrays['lon']

            # calculate IVT
            ivt_u = calculate_pressure_integral(qv*u,pressure*100)
            ivt_v = calculate_pressure_integral(qv*v,pressure*100)
            ivt = np.sqrt(ivt_u**2 + ivt_v**2)

            # add it to the output
            arrays['IVT'] = ivt

            # TODO: mask out low-latitude values?

            # return the dataset
            return out_mesh
        return execute_callback
