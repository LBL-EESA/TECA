import sys
import teca_py
import numpy as np


class teca_temporal_reduction_internals:
    class time_point:
        """
        A structure holding a floating point time value and its
        corresponding year, month day, hour minute and second
        """
        def __init__(self, t, units, calendar):
            self.t = t
            self.units = units
            self.calendar = calendar

            self.year, self.month, self.day, \
                self.hour, self.minutes, self.seconds = \
                teca_py.calendar_util.date(t, self.units, self.calendar)

        def __str__(self):
            return '%g (%s, %s) --> %04d-%02d-%02d %02d:%02d:%02g' % (
                self.t, self.units, self.calendar, self.year, self.month,
                self.day, self.hour, self.minutes, self.seconds)

    class c_struct:
        """
        A c like data structure
        """
        def __init__(self, **kwds):
            self.__dict__.update(kwds)

        def __str__(self):
            strg = ''
            for k, v in self.__dict__.items():
                strg += k + '=' + str(v) + ', '
            return strg

    class interval_iterator:
        class month_iterator:
            """
            An iterator over all months between 2 time_point's. A pair
            of time steps bracketing the current month are returned at
            each iteration.
            """

            def __init__(self, t, units, calendar):
                """
                t - an array of floating point time values
                units - string units of the time values
                calendar - string name of the calendar system
                """
                self.t = t
                self.units = units

                calendar = calendar.lower()
                self.calendar = calendar

                # time point's to iterate between
                self.t0 = teca_temporal_reduction_internals.time_point(
                              t[0], units, calendar)

                self.t1 = teca_temporal_reduction_internals.time_point(
                              t[-1], units, calendar)

                # current time state
                self.year = self.t0.year
                self.month = self.t0.month

            def last_day_of_month(self):
                """
                get the number of days in the month, with logic for
                leap years
                """
                return \
                    teca_py.calendar_util.days_in_month(self.calendar,
                                                        self.units, self.year,
                                                        self.month)

            def __iter__(self):
                return self

            def __next__(self):
                """
                return a pair of time steps bracketing the current month.
                both returned time steps belong to the current month.
                """
                # check for more months to process
                if (self.year > self.t1.year) or \
                        (self.year == self.t1.year) and \
                        (self.month > self.t1.month):
                    raise StopIteration

                # find the time step of the first day
                year = self.year
                month = self.month

                t0 = '%04d-%02d-01 00:00:00' % (self.year, self.month)
                i0 = teca_py.coordinate_util.time_step_of(self.t, True, True,
                                                          self.calendar,
                                                          self.units, t0)

                # find the time step of the last day
                n_days = self.last_day_of_month()

                t1 = '%04d-%02d-%02d 23:59:59' % \
                    (self.year, self.month, n_days)

                i1 = teca_py.coordinate_util.time_step_of(self.t, True, True,
                                                          self.calendar,
                                                          self.units, t1)

                # move to next month
                self.month += 1

                # move to next year
                if self.month == 13:
                    self.month = 1
                    self.year += 1

                return teca_temporal_reduction_internals.c_struct(
                    time=self.t[i0], year=year, month=month,
                    day=1, start_index=i0, end_index=i1)

        class day_iterator:
            """
            An iterator over all days between 2 time_point's. A pair
            of time steps bracketing the current day are returned at
            each iteration.
            """

            def __init__(self, t, units, calendar):
                """
                t - an array of floating point time values
                units - string units of the time values
                calendar - string name of the calendar system
                """
                # time values
                self.t = t
                self.units = units

                calendar = calendar.lower()
                self.calendar = calendar

                # time point's to iterate between
                self.t0 = teca_temporal_reduction_internals.time_point(
                              t[0], units, calendar)

                self.t1 = teca_temporal_reduction_internals.time_point(
                              t[-1], units, calendar)

                # current time state
                self.year = self.t0.year
                self.month = self.t0.month
                self.day = self.t0.day

            def last_day_of_month(self):
                """
                get the number of days in the month, with logic for
                leap years
                """
                return teca_py.calendar_util.days_in_month(
                           self.calendar, self.units, self.year, self.month)

            def __iter__(self):
                return self

            def __next__(self):
                """
                return a pair of time steps bracketing the current month.
                both returned time steps belong to the current month.
                """
                # check for more days to process
                if (self.year > self.t1.year) or \
                        ((self.year == self.t1.year) and
                         (self.month > self.t1.month)) or \
                        ((self.year == self.t1.year) and
                         (self.month == self.t1.month) and
                         (self.day > self.t1.day)):
                    raise StopIteration

                # find the time step of the first day
                year = self.year
                month = self.month
                day = self.day

                t0 = '%04d-%02d-%02d 00:00:00' % \
                    (self.year, self.month, self.day)

                i0 = teca_py.coordinate_util.time_step_of(self.t, True, True,
                                                          self.calendar,
                                                          self.units, t0)

                # find the time step of the last day
                t1 = '%04d-%02d-%02d 23:59:59' % \
                    (self.year, self.month, self.day)

                i1 = teca_py.coordinate_util.time_step_of(self.t, True, True,
                                                          self.calendar,
                                                          self.units, t1)

                # move to next day
                n_days = self.last_day_of_month()
                self.day += 1

                # move to next month
                if self.day > n_days:
                    self.month += 1
                    self.day = 1

                # move to next year
                if self.month == 13:
                    self.month = 1
                    self.year += 1

                return teca_temporal_reduction_internals.c_struct(
                    time=self.t[i0], year=year, month=month, day=day,
                    start_index=i0, end_index=i1)

        @staticmethod
        def New(interval, t, units, calendar):
            if interval == 'monthly':

                return teca_temporal_reduction_internals. \
                    interval_iterator.month_iterator(t, units, calendar)

            elif interval == 'daily':

                return teca_temporal_reduction_internals. \
                    interval_iterator.day_iterator(t, units, calendar)

            else:

                raise RuntimeError('Invlid interval %s' % (interval))

    class reduction_operator:
        class average:
            num_t = []

            def __init__(self):
                self.count = 1.0

            def update(self, out_array, in_array):
                # track number of entries for average.
                self.count += 1
                # don't use integer types for this calculation
                if in_array.dtype.kind == 'i':
                    in_array = in_array.astype(np.float32) \
                        if in_array.itemsize < 8 else \
                        in_array.astype(float64)

                if out_array.dtype.kind == 'i':
                    out_array = out_array.astype(np.float32) \
                        if out_array.itemsize < 8 else \
                        out_array.astype(float64)

                # accumulate
                return out_array + in_array

            def finalize(self, out_array):
                n = self.count
                self.count = 1.0
                return out_array / n

        class minimum:
            def update(self, out_array, in_array):
                return np.minimum(out_array, in_array)

            def finalize(self, out_array):
                return out_array

        class maximum:
            def update(self, out_array, in_array):
                return np.maximum(out_array, in_array)

            def finalize(self, out_array):
                return out_array

        @staticmethod
        def New(op_name):
            if op_name == 'average':
                return teca_temporal_reduction_internals. \
                    reduction_operator.average()

            elif op_name == 'minimum':
                return teca_temporal_reduction_internals. \
                    reduction_operator.minimum()

            elif op_name == 'maximum':
                return teca_temporal_reduction_internals. \
                    reduction_operator.maximum()

            raise RuntimeError('Invalid operator %s' % (op_name))


class teca_temporal_reduction(teca_py.teca_threaded_python_algorithm):
    """
    Reduce a mesh across the time dimensions by a defined increment using
    a defined operation.

        time increments: daily, monthly
        reduction operators: average, min, max

    The output time axis will be defined using the selected increment.
    The output data will be accumulated/reduced using the selected
    operation.
    """
    def __init__(self):
        self.indices = []
        self.arrays = []
        self.interval_name = None
        self.operator_name = None
        self.operator = {}

    def set_interval(self, interval):
        """
        set the output interval
        """
        self.interval_name = interval

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

    def set_operator(self, operator):
        """
        set the reduction operator
        """
        self.operator_name = operator

    def set_operator_to_maximum(self):
        """
        set the reduction operator to maximum.
        """
        self.operator_name = 'maximum'

    def set_operator_to_minimum(self):
        """
        set the reduction operator to minimum.
        """
        self.operator_name = 'minimum'

    def set_operator_to_average(self):
        """
        set the reduction operator to average.
        """
        self.operator_name = 'average'

    def set_arrays(self, arrays):
        """
        Set the list of arrays to reduce
        """
        if isinstance(arrays, list):
            arrays = list(arrays)
        self.arrays = arrays

    def report(self, port, md_in):
        """
        implements the report phase of pipeline execution
        """
        if self.get_verbose() > 0:
            try:
                rank = self.get_communicator().Get_rank()
            except Exception:
                rank = 0
            sys.stderr.write('[%d] teca_temporal_reduction::report\n' % (rank))

        # sanity checks
        if self.interval_name is None:
            raise RuntimeError('No interval specified')

        if self.operator_name is None:
            raise RuntimeError('No operator specified')

        if self.arrays is None:
            raise RuntimeError('No arrays specified')

        md_out = md_in[0]

        # get the input time axis
        atts = md_out['attributes']
        coords = md_out['coordinates']

        t = coords['t']
        t_var = coords['t_variable']

        t_atts = atts[t_var]
        cal = t_atts['calendar']
        t_units = t_atts['units']

        # convert the time axis to a monthly delta t
        self.indices = [ii for ii in teca_temporal_reduction_internals.
                        interval_iterator.New(
                            self.interval_name, t, t_units, cal)]

        if self.get_verbose() > 1:
            sys.stderr.write('indices = [\n')
            for ii in self.indices:
                sys.stderr.write('\t%s\n' % (str(ii)))
            sys.stderr.write(']\n')

        # update the pipeline control keys
        initializer_key = md_out['index_initializer_key']
        md_out[initializer_key] = len(self.indices)

        # update the metadata so that  modified time axis and reduced variables
        # are presented
        out_atts = teca_py.teca_metadata()
        out_vars = []

        for array in self.arrays:
            # name of the output array
            out_vars.append(array)

            # pass the attributes
            in_atts = atts[array]

            # convert integer to floating point for averaging operations
            if self.operator_name == 'average':
                tc = in_atts['type_code']
                if tc == teca_py.teca_int_array_code.get()                  \
                        or tc == teca_py.teca_char_array_code.get()         \
                        or tc == teca_py.teca_unsigned_int_array_code.get() \
                        or tc == teca_py.teca_unsigned_char_array_code.get():
                    tc = teca_py.teca_float_array_code.get()
                elif tc == teca_py.teca_long_long_array_code.get()          \
                        or tc == teca_py.teca_unsigned_long_long_array_code.get():
                    tc = teca_py.teca_double_array_code.get()
                in_atts['type_code'] = tc

            # document the transformation
            in_atts['description'] = '%s %s of %s' % (self.interval_name,
                                                      self.operator_name,
                                                      array)

            out_atts[array] = in_atts

        # update time axis
        q = 0
        t_out = np.empty(len(self.indices), dtype=np.float64)
        for ii in self.indices:
            t_out[q] = ii.time
            q += 1
        coords['t'] = t_out
        md_out['coordinates'] = coords

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
                rnk = self.get_communicator().Get_rank()
            except Exception:
                rnk = 0
            sys.stderr.write('[%d] teca_temporal_reduction::request\n' % (rnk))

        # initialize a new reduction operator, for the subsequent
        # execute
        for array in self.arrays:
            self.operator[array] = teca_temporal_reduction_internals. \
                reduction_operator.New(self.operator_name)

        # generate one request for each time step in the interval
        up_reqs = []

        md = md_in[0]
        request_key = md['index_request_key']
        req_id = req_in[request_key]
        ii = self.indices[req_id]
        i = ii.start_index
        while i <= ii.end_index:
            req = teca_py.teca_metadata(req_in)
            req[request_key] = i
            up_reqs.append(req)
            i += 1

        return up_reqs

    def execute(self, port, data_in, req_in, streaming):
        """
        implements the execute phase of pipeline execution
        """

        # get the requested index and its
        request_key = req_in['index_request_key']
        req_id = req_in[request_key]
        ii = self.indices[req_id]

        if self.get_verbose() > 0:
            try:
                rank = self.get_communicator().Get_rank()
            except Exception:
                rank = 0
            sys.stderr.write('[%d] teca_temporal_reduction::execute '
                             'request %d (%d - %d), reducing %d, %d '
                             'remain\n' % (rank, req_id, ii.start_index,
                                           ii.end_index, len(data_in),
                                           streaming))

        # copy the first mesh
        mesh_in = teca_py.as_teca_cartesian_mesh(data_in.pop())
        mesh_out = teca_py.teca_cartesian_mesh.New()
        mesh_out.copy(mesh_in)
        arrays_out = mesh_out.get_point_arrays()

        # accumulate incoming values
        while len(data_in):
            mesh_in = teca_py.as_teca_cartesian_mesh(data_in.pop())
            arrays_in = mesh_in.get_point_arrays()
            for array in self.arrays:
                arrays_out[array] = \
                    self.operator[array].update(arrays_out[array],
                                                arrays_in[array])

        # when all the data is processed
        if not streaming:
            # finalize reduction
            for array in self.arrays:
                arrays_out[array] = \
                    self.operator[array].finalize(arrays_out[array])

            # fix time
            mesh_out.set_time_step(req_id)
            mesh_out.set_time(ii.time)

        return mesh_out
