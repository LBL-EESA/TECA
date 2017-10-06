import numpy as np
import teca_py
import sys

class teca_ar_rutz(object):
    """ """
    @staticmethod
    def New():
        return teca_ar_rutz()

    def __init__(self):

        self.arrays = ['IVT']
        self.array_dict = {}
        self.bounds = []

        self.impl = teca_py.teca_programmable_algorithm.New()
        self.impl.set_number_of_input_connections(1)
        self.impl.set_number_of_output_ports(1)
        self.impl.set_report_callback(self.get_report_callback())
        self.impl.set_request_callback(self.get_request_callback())
        self.impl.set_execute_callback(self.get_execute_callback())


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
        """ Returns a proper TECA report callback function """
        def report_callback(port, md_in):
            """ Defines the variables that this class returns. """
            # add the names of the variables this method generates
            md_out = teca_py.teca_metadata(md_in[0])
            md_out.append('variables', 'AR')
            return md_out
        
        return report_callback

    def get_request_callback(self):
        """ Returns a proper TECA request callback function """
        def request_callback(port, md_in, req_in):
            """ Requests the variables needed to find ARs """

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

            # add the arrays we need
            req_out['arrays'] = self.arrays + req_out['arrays'] if \
		req_out.has('arrays') else self.arrays

            sys.stderr.write('\n=================teca_ar_rutz request_callback\n%s\n'%(str(req_out)))
            reqs_out = [req_out]
            return reqs_out
        return request_callback

    def get_execute_callback(self):
        """ Returns a proper TECA execute callback function """
        def execute_callback(port, data_in, req_in):

            # get the patch
            mesh = teca_py.as_teca_cartesian_mesh(data_in[0])

	    # maybe no mesh if there are more ranks than time steps
            if mesh is None:
                return teca_py.teca_table.New()

            # coordinates
            lon = mesh.get_x_coordinates().as_array()
            lat = mesh.get_y_coordinates().as_array()
	    lev = mesh.get_z_coordinates().as_array()

            time = mesh.get_time()
            step = mesh.get_time_step()

            nx = len(lon)
            ny = len(lat)
	    nz = len(lev)

            sys.stderr.write('\n=================teca_ar_rutz execute_callback\n%s\n'%(str(self.arrays)))
            for name in self.arrays:
                # get the named scalar
                array = mesh.get_point_arrays().get(name).as_array()
                # convert from a flat 1D array into a 2D array
                array = np.reshape(array, [nz,ny,nx])
		# do something here...
		sys.stderr.write('got array %s = %s\n'%(name, str(array)))

            # make the event table
            events = teca_py.teca_table.New()
            # do something here ...
            return events

        return execute_callback
