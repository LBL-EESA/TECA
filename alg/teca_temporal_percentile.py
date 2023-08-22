import sys
import numpy
if get_teca_has_cupy():
    import cupy

class teca_temporal_percentile(teca_python_algorithm):
    """
    Reduce a mesh across the time dimensions by a defined increment using
    a percentile.

        time increments: daily, monthly, seasonal

    The output time axis will be defined using the selected increment.
    The output data will be accumulated/reduced using the selected
    operation.
    """
    class interval_iterator_collection:
        """
        A collection of interval_iterators compatible with
        teca_temporal_percentile and a factory method that creates ionstances
        from a string that names the type.

        This collectiond implements the following interval_iterators

            daily
            monthly
            seasonal
            yearly
            n_steps

        The factory method (interval_iterator_collection.New) retruns instances
        when passed a string naming one of the above iterators. For the n_steps
        iterator replace n with the desired number of steps (eg. 8_steps)
        """
        def New(self, interval, t, units, calendar):
            if interval == 'seasonal':

                return teca_temporal_reduction. \
                    interval_iterator_collection. \
                        season_iterator(t, units, calendar)

            if interval == 'monthly':

                return teca_temporal_reduction. \
                    interval_iterator_collection. \
                        month_iterator(t, units, calendar)

            elif interval == 'daily':

                return teca_temporal_reduction. \
                    interval_iterator_collection. \
                        day_iterator(t, units, calendar)

            elif (pos := interval.rfind('_steps')) > 0:

                n_steps = int(interval[0:pos])

                return teca_temporal_reduction. \
                    interval_iterator_collection. \
                        n_steps_iterator(t, units, calendar, n_steps)

            else:

                raise RuntimeError('Invlid interval %s' % (interval))


    def __init__(self):
        self.have_cuda = get_teca_has_cuda() and get_teca_has_cupy()
        self.indices = []
        self.point_arrays = []
        self.interval_name = None
        self.percentile = None
        self.use_fill_value= True

        self.interval_iterator_factory = \
            teca_temporal_percentile.interval_iterator_collection()

    def set_interval_iterator_factory(self, factory):
        """
        Sets a factory object that implelents a method named New that,
        given a string naming the interval, can create the coresponding
        interval_iterator. The factory has the following signature:

            New(self, interval_name, t, units, calendar) -> interval_iterator

        where

            interval_name - a string naming the type of interval iterator to
                          - create
            t             - an array with the time coordinates to iterate over
            units         - time units
            calendar      - the calendar
        """
        self.interval_iterator_factory = factory

    def set_percentile(self, percentile):
        """
        set the percentile to compute between 0 and 100.
        """
        self.percentile = percentile

    def set_interval(self, interval):
        """
        set the output interval
        """
        self.interval_name = interval

    def set_interval_to_seasonal(self):
        """
        set the output interval to seasonal.
        """
        self.interval_name = 'seasonal'

    def set_interval_to_monthly(self):
        """
        set the output interval to monthly.
        """
        self.interval_name = 'monthly'

    def set_interval_to_daily(self):
        """
        set the output interval to daily.
        """
        self.interval_name = 'daily'

    def set_interval_to_n_steps(self, n_steps):
        """
        set the output interval to n_steps.
        """
        self.interval_name = '%d_steps'%(n_steps)

    def set_point_arrays(self, arrays):
        """
        Set the list of arrays to reduce
        """
        if isinstance(arrays, list):
            arrays = list(arrays)
        self.point_arrays = arrays

    def set_use_fill_value(self, use_fill_value):
        """
        if set will disable missing value handling
        """
        self.use_fill_value = use_fill_value

    @staticmethod
    def get_fill_value(array, atts):
        """
        Given an array attribute collection look for the fill value in the
        usual locations
        """
        if atts.has('_FillValue'):
            return atts['_FillValue']
        elif atts.has('missing_value'):
            return atts['missing_value']
        raise RuntimeError('Failed to determine the fill value for %s'%(array))

    def report(self, port, md_in):
        """
        implements the report phase of pipeline execution
        """
        if self.get_verbose() > 0:
            try:
                rank = self.get_communicator().Get_rank()
            except Exception:
                rank = 0
            sys.stderr.write('[%d] teca_temporal_percentile::report\n' % (rank))

        # sanity checks
        if self.interval_name is None:
            raise RuntimeError('No interval specified')

        if self.point_arrays is None:
            raise RuntimeError('No arrays specified')

        md_out = md_in[0]

        # get the input time axis and metadata
        atts = md_out['attributes']
        coords = md_out['coordinates']

        t = coords['t']
        t_var = coords['t_variable']
        t_atts = atts[t_var]

        try:
            cal = t_atts['calendar']
        except KeyError:
            cal = 'standard'
            sys.stderr.write('Attributes for the time axis %s is missing '
                             'calendar. The "standard" calendar will be '
                             'used'%(t_var))

        t_units = t_atts['units']

        # convert the time axis to the specified interval
        self.indices = [ii for ii in self.interval_iterator_factory. \
                        New(self.interval_name, t, t_units, cal)]

        if self.get_verbose() > 1:
            sys.stderr.write('indices = [\n')
            for ii in self.indices:
                sys.stderr.write('\t%s\n' % (str(ii)))
            sys.stderr.write(']\n')

        # update the pipeline control keys
        initializer_key = md_out['index_initializer_key']
        md_out[initializer_key] = len(self.indices)

        # update the metadata so that modified time axis and reduced variables
        # are presented
        out_atts = teca_metadata()
        out_vars = []

        for array in self.point_arrays:
            # name of the output array
            out_vars.append(array)

            # pass the attributes
            atts = atts[array]

            # document the transformation
            atts['description'] = '%s %dth percentile of %s' % (
                self.interval_name, self.percentile, array)

            out_atts[array] = atts

        # update time axis
        t_out = np.empty(len(self.indices), dtype=np.float64)

        q = 0
        for ii in self.indices:
            t_out[q] = ii.time
            q += 1

        coords['t'] = t_out
        md_out['coordinates'] = coords

        # update the attributes
        out_atts[t_var] = t_atts

        # package it all up and return
        md_out['variables'] = out_vars
        md_out["attributes"] = out_atts

        return md_out

    def request(self, port, md_in, req_in):
        """
        implements the request phase of pipeline execution
        """
        if self.get_verbose() > 0:
            try:
                rank = self.get_communicator().Get_rank()
            except Exception:
                rank = 0
            sys.stderr.write('[%d] teca_temporal_percentile::request\n' % (rank))

        md = md_in[0]

        # get the available arrays
        vars_in = []
        if md.has('variables'):
            vars_in = md['variables']
            if not isinstance(vars_in, list):
                vars_in = [vars_in]

        # get the requested arrays
        req_arrays = []
        if req_in.has('arrays'):
            req_arrays = req_in['arrays']
            if not isinstance(req_arrays, list):
                req_arrays = [req_arrays]

        # get the array attributes
        atrs = md['attributes']

        for array in self.point_arrays:

            # request the array
            if array not in req_arrays:
                req_arrays.append(array)

            # request the associated valid value mask
            if self.use_fill_value:
                vv_mask = array + '_valid'
                if (vv_mask in vars_in) \
                    and (vv_mask not in req_arrays):
                    req_arrays.append(vv_mask)

        # generate a request for the range of time steps in the interval
        up_reqs = []

        request_key = md['index_request_key']
        req_id = req_in[request_key]

        ii = self.indices[req_id[0]]

        req = teca_metadata(req_in)
        req['arrays'] = req_arrays
        req[request_key] = [ii.start_index, ii.end_index]
        up_reqs.append(req)

        return up_reqs

    def execute(self, port, data_in, req_in):
        """
        implements the execute phase of pipeline execution
        """
        # get the requested index
        request_key = req_in['index_request_key']
        req_id = req_in[request_key]
        ii = self.indices[req_id[0]]

        # get the device to execute on
        dev = -1
        if self.have_cuda:
            dev = req_in['device_id']

        if self.get_verbose() > 0:
            try:
                rank = self.get_communicator().Get_rank()
            except Exception:
                rank = 0
            sys.stderr.write('[%d] teca_temporal_percentile::execute '
                             'indices: %d-%d device: %s\n' % ( rank,
                             ii.start_index, ii.end_index,
                             'CPU' if dev < 0 else 'GPU %d'%(dev)) )

        # get the input mesh
        mesh_in = as_teca_cartesian_mesh(data_in[0])

        # create the output mesh, pass metadata
        mesh_out = teca_cartesian_mesh.New()
        mesh_out.copy_metadata(mesh_in)

        arrays_out = mesh_out.get_point_arrays()

        # get the collection of point centered arrays and attributes
        arrays_in = mesh_in.get_point_arrays()
        atrs = mesh_in.get_attributes()

        # reduce each array
        for array in self.point_arrays:

            # get the shape of the array
            nx,ny,nz,nt = mesh_in.get_array_shape(array)

            # check for valid value masks indicating the presence of missing
            # values. Cupy's implementation does not handle missing values so
            # if missing values are present fallback to numpy
            valid = array + '_valid'
            have_vvm = self.use_fill_value and arrays_in.has(valid)

            if dev < 0 or have_vvm:
                # Compute the percentile on the CPU
                array_in = arrays_in[array].get_host_accessible()
                array_in.shape = (nt,nz,ny,nx)

                if have_vvm:
                    # get the input array on the CPU
                    valid_in = arrays_in[valid].get_host_accessible()
                    valid_in.shape = (nt,nz,ny,nx)

                    # convert the missing values to NaN as required by Numpy
                    array_in = np.where(valid_in, array_in, np.nan)

                    # compute the percentile with missing value handling
                    array_out = numpy.nanpercentile(array_in, self.percentile, axis=0)

                    # convert NaN back to the missing value value
                    nans = np.isnan(array_out)
                    array_out[nans] = self.get_fill_value(array, atrs[array])
                else:
                    # compute percentile on the CPU without missing value handling
                    array_out = numpy.percentile(array_in, self.percentile, axis=0)

            else:
                # compute the percentile on the GPU without missing value handling
                cupy.cuda.Device(dev).use()

                array_in = arrays_in[array].get_cuda_accessible()
                array_in.shape = (nt,nz,ny,nx)

                array_out = cupy.percentile(array_in, self.percentile, axis=0)

            # save the result. Numpy uses float64 so a type conversion may be needed
            arrays_out[array] = array_out.astype( array_in.dtype )

        # fix time
        mesh_out.set_time_step(req_id[0])
        mesh_out.set_time(ii.time)

        return mesh_out
