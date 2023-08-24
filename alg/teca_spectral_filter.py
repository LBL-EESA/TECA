import sys, time
import argparse
import numpy
if get_teca_has_cuda() and get_teca_has_cupy():
    import cupy


class teca_spectral_filter(teca_python_algorithm):
    """ Apply a low or high-pass Butterworth filter point-wise. The window size
    should be set to the entire time domain to avoid edge affects. Use spatial
    parallelism and more MPI ranks for larger data.  """

    def __init__(self):
        """ initialize the class with defaults """

        self.have_cuda = get_teca_has_cuda() and get_teca_has_cupy()
        self.filter_type = 'low_pass'
        self.filter_order = 0
        self.critical_frequency = 0.
        self.point_arrays = []
        self.pass_input_arrays = True

    def set_filter_parameters(self, ftype, order, period, units):
        """ set the filter type, order, and critical frequency. The frequnecy
        is computed from a period and units specification. The filter type can
        be low_pass or high_pass. The units can be days, hours, minutes, or
        seconds """

        if ftype != 'low_pass' and ftype != 'high_pass':
            raise ValueError('Invalid filter type %s. The filter type may be '
                             'low_pass or high_pass') % (ftype)

        self.filter_type = ftype
        self.filter_order = order
        self.critical_frequency = 1. / ( period * self.get_number_of_seconds(units) )

    def set_filter_parameters_hz(self, ftype, order, fcrit):
        """ set the filter type, order, and critical frequency in Hz. The
        filter type can be low_pass or high_pass. """

        if ftype != 'low_pass' and ftype != 'high_pass':
            raise ValueError('Invalid filter type %s. The filter type may be '
                             'low_pass or high_pass') % (ftype)

        self.filter_type = ftype
        self.filter_order = order
        self.critical_frequency = fcrit

    def set_pass_input_arrays(self, pass_in):
        """ When set to True the input arrays will be included in the output.
        In that case the filtered array will be named: <name>_<filter_type>
        where name is the name of the input array and filter type is one of
        low_pass or high_pass """

        self.pass_input_arrays = pass_in

    def set_point_arrays(self, arrays):
        """ Set the list of arrays to filter """

        if not isinstance(arrays, list):
            arrays = list(arrays)

        self.point_arrays = arrays

    @staticmethod
    def get_number_of_seconds(unit):
        """ get the number of seconds in the specified unit. unit string must
        contaoin one of seconds, minutes, hours, or days. A ValueError is
        raised when invalid units are encountered """

        if 'seconds' in unit:
            return 1.
        elif 'minutes' in unit:
            return 60.
        elif 'hours' in  unit:
            return 60.*60.
        elif 'days' in unit:
            return 60.*60.*24.

        raise ValueError('Unsupported time axis units %s' % (unit))

    @staticmethod
    def butterworth(npmod, n_samples, sample_rate, critical_freq, order, high_pass):
        """ Generate the frequency response for a Butterworth filter of the given
        order.

        npmod - the compute module (numpy or cupy)
        n_samples - filter size in number of samples
        sample_rate - sample rate in Hz
        critical_freq - cut off frequency in Hz
        order - filter order
        high_pass - if true the high pass version is generated """
        fs2 = sample_rate / 2.
        f = npmod.linspace(-fs2, fs2, n_samples, dtype=npmod.float64)
        o2 = -2.*order if high_pass else 2.*order
        h =  1. / npmod.sqrt( 1. + npmod.power( f / critical_freq, o2 ) )
        return h

    @staticmethod
    def butterworth_ideal(npmod, n_samples, sample_rate, critical_freq, high_pass):
        """ Generate the frequency response for a Butterworth filter of
        infinite order.

        npmod - the compute module (numpy or cupy)
        n_samples - filter size in number of samples
        sample_rate - sample rate in Hz
        critical_freq - cut off frequency in Hz
        order - filter order
        high_pass - if true the high pass version is generated """
        fs2 = sample_rate / 2.
        f = npmod.linspace(-fs2, fs2, n_samples, dtype=npmod.float64)
        h = npmod.zeros(n_samples, dtype=npmod.float64)
        modf = npmod.abs(f)
        if high_pass:
            mask = npmod.where(modf > critical_freq, True, False)
        else:
            mask = npmod.where(modf < critical_freq, True, False)
        h[mask] = 1.
        return h

    def report(self, port, rep_in):
        """ TECA report override """
        # copy the incoming report
        rep = rep_in[0]

        # get the attributes collection
        attributes = rep["attributes"]

        # for each array to filter
        for array_in in self.point_arrays:

            # add filter parameters to the description
            array_atts = attributes[array_in]

            descr = array_atts['description'] if array_atts.has('descritpion') else ''
            array_atts['description'] = descr + '(%s filtered f_crit = %g)' % (
                                        self.filter_type, self.critical_frequency)

            # rename if passing the input through
            array_out = array_in
            if self.pass_input_arrays:
                array_out += '_' + self.filter_type

            # update the array attributes
            attributes[array_out] = array_atts

        # update the attributes collection
        rep['attributes'] = attributes

        return rep

    def request(self, port, md_in, req_in):
        """ TECA request override """

        # cpoy the incoming request to preserve down stream requirements
        req = teca_metadata(req_in)

        # get the list of requested arrays
        arrays = req['arrays']
        if not isinstance(arrays, list):
            arrays = list(arrays)

        # for each array to filter
        for array_in in self.point_arrays:

            # remove the result array from the request
            if self.pass_input_arrays:
                array_out = array_in + '_' + self.filter_type
                if array_out in arrays:
                    arrays.remove(array_out)

            # request the array to filter
            if not array_in in arrays:
                arrays.append(array_in)

        # update the list of requested arrays
        req['arrays'] = arrays

        return [req]

    def execute(self, port, data_in, req):
        """ TECA execute override """
        try:
            rank = self.get_communicator().Get_rank()
        except Exception:
            rank = 0

        # get the device to execute on, assigned by the execution engine
        dev = -1
        npmod = numpy
        if self.have_cuda:
            dev = req['device_id']
            if dev >= 0:
                npmod = cupy
                cupy.cuda.Device(dev).use()

        # get the input
        mesh_in = as_teca_cartesian_mesh(data_in[0])

        # allocate the output and shallow copy to preserve data that we don't process
        mesh_out = teca_cartesian_mesh.New()
        mesh_out.shallow_copy(mesh_in)

        # get the time units in seconds
        t_units = mesh_in.get_time_units()
        seconds_per = self.get_number_of_seconds(t_units)

        # get the number of time steps and time span of this window
        t_ext = mesh_in.get_temporal_extent()
        win_size = t_ext[1] - t_ext[0] + 1

        t_bds = mesh_in.get_temporal_bounds()
        t_win = t_bds[1] - t_bds[0]

        # compute the sampling rate in Hz
        sample_rate = win_size / ( t_win * seconds_per )

        # generate the filter kernel
        high_pass = 1 if self.filter_type == 'high_pass' else 0

        if self.filter_order < 1:
            H = self.butterworth_ideal(npmod, win_size, sample_rate,
                                       self.critical_frequency, high_pass)
        else:
            H = self.butterworth(npmod, win_size, sample_rate,
                                 self.critical_frequency, self.filter_order,
                                 high_pass)

        # reshape so that numpy broadcasting works
        H.shape = [win_size, 1, 1, 1]

        # apply the filter in the frequenmcy domain
        arrays_in = mesh_in.get_point_arrays()
        arrays_out = mesh_out.get_point_arrays()

        for array_name_in in self.point_arrays:
            wct_0 = time.monotonic_ns()

            # get the input array
            if dev < 0:
                array_in = arrays_in[array_name_in].get_host_accessible()
            else:
                array_in = arrays_in[array_name_in].get_cuda_accessible()

            # reshape the input array
            nx,ny,nz,nt = mesh_in.get_array_shape(array_name_in)
            array_in.shape = (nt,nz,ny,nx)

            # apply the filter in the frequency domain
            array_out = npmod.fft.ifft( npmod.fft.ifftshift(
                            npmod.fft.fftshift( npmod.fft.fft(
                                array_in , axis=0 ), axes=0) * H,
                                    axes=0 ), axis=0 )

            # store in the output
            array_name_out = array_name_in + '_' + self.filter_type

            arrays_out[array_name_out] = npmod.ravel( npmod.array(
                array_out.real, copy=True, dtype=array_in.dtype ) )

            # report what was done
            if self.get_verbose():
                nc = nx*ny*nz
                wct_1 = time.monotonic_ns()
                sys.stderr.write('[%d] STATUS: teca_spectral_filter::execute %s '
                                 'win_size=%d sample_rate=%g f_crit=%g order=%.0f '
                                 'n_cells=%d dev=%s completed in %.3f sec\n' % (rank,
                                 array_name_in, win_size, sample_rate,
                                 self.critical_frequency, self.filter_order,
                                 nc, 'CPU' if dev < 0 else 'GPU %d' % (dev),
                                 (wct_1 - wct_0) / 1.0e9))

        return mesh_out
