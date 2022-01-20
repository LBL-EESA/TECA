import sys
import numpy as np

class teca_temporal_reduction(teca_threaded_python_algorithm):
    """
    Reduce a mesh across the time dimensions by a defined increment using
    a defined operation.

        time increments: daily, monthly, seasonal
        reduction operators: average, min, max

    The output time axis will be defined using the selected increment.
    The output data will be accumulated/reduced using the selected
    operation.

    The set_use_fill_value  method controls how invalid or missing values are
    teated.  When set to 1, NetCDF CF fill values are detected and handled.
    This is the default. If it is known that the dataset has no invalid or
    missing values one may set this to 0 for faster processing. By default the
    fill value will be obtained from metadata stored in the NetCDF CF file
    (_FillValue). One may override this by explicitly calling set_fill_value
    method with the desired fill value.

    For minimum and maximum operations, at given grid point only valid values
    over the interval are used in the calculation.  if there are no valid
    values over the interval at the grid point it is set to the fill_value.

    For the averaging operation, during summation missing values are treated
    as 0.0 and a per-grid point count of valid values over the interval is
    maintained and used in the average. Grid points with no valid values over
    the inteval are set to the fill value.

    User defined reductions:
    ------------------------
    A reduction_operator compatible with teca_temporal_reduction must implement
    3 class methods: initialize, update, and finalize.

        initialize(self, fill_value) -> None

            initializes the reduction. If a not None fill value is passed
            the operator should use it to identify missing values in the data
            and handle them approproiately.

        update(self, out_array, in_array) -> numpy ndarray

            reduces in_array (new data) into out_array (current state) and
            returns the result.

        finalize(self, out_array) -> numpy ndarray

            finalizes out_array (current state) and returns the result.
            if no finalization is needed simpy return out_array.

    A reduction_operator_factory compatible with teca_temporal_reduction
    must implement a factory method named New that takes a string and returns
    a reduction_operator.

        New(self, op_name) -> reduction_operator

        A factory method that creates an instance on demand from a string that
        matches its name. The existing operator_collection may be extended by
        overriding this method and falling back to the base method when the
        passed string is unknown.

    To use a user defined custom reduction operator, one must install the
    factory that creates it by passing a factory instance to
    teca_temporal_reduction.set_reduction_operator_factory.

    User defined time intervals:
    ----------------------------
    An interval_iterator compatible with teca_temporal_reduction must implement
    the Python iterator methods : __init__, __iter__ and __next__. The __init__
    method will be passed a floating point array of time values to iterate over
    and calendaring metadata (calendar name and time units strings) needed to
    interpret the values.  The __iter__method retuns self. The __next__ method
    determines the indices that span the next interval and return the start and
    end index into the time array as well as the floating point time of the
    start of the interval.  Following the Python iterator protocol __next_
    raises a StopIteration exception when all intervals have been visited and
    iteration should stop. The __next__ method will return a time_interval
    object. This object must have the following public member variables: time,
    start_index, and end_index.

        __init__(self, t, units, calendar)

            initializes the interval iterator from time values in t and
            calendaring metadata in units and calendar

        __iter__(self) -> self

            Boiler plate Python iterator protocol

        __next__(self) -> time_interval

            returns the next time_interval in the series and raises
            StopIteration when the series in complete.

    An interval_iterator_factory compatible with teca_temporal_reduction
    must implement a factory method named New that takes a string naming
    the type of the interval iterator to create, the floating point time
    values to iterate over, and calendaring metadata. The factory will
    return an iterator instance or raise a RuntimeError if no such interval
    iterator is defined.

        New(self, it_name, time_vals, units, calendar) -> interval_iterator

        A factory method that creates an instance on demand from a string that
        matches its name. The existing iterator_collection may be extended by
        overriding this method and falling back to the base method when the
        passed string is unknown.
    """

    class reduction_operator_collection:
        """
        A collection of reduction_operators compatible with
        teca_temporal_reduction, and a factory method that creates one on demand
        from a runtime provided string.

        This collection implements the following operators:

            minimum
            maximum
            average
            summation

        The factory method (operator_collection.New) retruns instances when passed
        a string naming one of them.
        """
        class average:
            def __init__(self):
                self.count = None
                self.fill_value = None

            def initialize(self, fill_value):
                self.fill_value = fill_value

            def update(self, out_array, in_array):
                # don't use integer types for this calculation
                if in_array.dtype.kind == 'i':
                    in_array = in_array.astype(np.float32) \
                        if in_array.itemsize < 8 else \
                        in_array.astype(float64)

                if out_array.dtype.kind == 'i':
                    out_array = out_array.astype(np.float32) \
                        if out_array.itemsize < 8 else \
                        out_array.astype(float64)

                # identify the invalid values
                if self.fill_value is not None:
                    out_is_bad = np.isclose(out_array, self.fill_value)
                    in_is_bad = np.isclose(in_array, self.fill_value)

                # initialize the count the first time through. this needs to
                # happen now since before this we don't know where invalid
                # values are.
                if self.count is None:
                    if self.fill_value is None:
                        self.count = 1.0
                    else:
                        self.count = np.where(out_is_bad, np.float32(0.0),
                                              np.float32(1.0))

                if self.fill_value is not None:
                    # update the count only where there is valid data
                    self.count += np.where(in_is_bad, np.float32(0.0),
                                           np.float32(1.0))

                    # accumulate
                    tmp = np.where(out_is_bad, np.float32(0.0), out_array) \
                        + np.where(in_is_bad, np.float32(0.0), in_array)

                else:
                    # update count
                    self.count += np.float32(1.0)

                    # accumulate
                    tmp = out_array + in_array

                return tmp

            def finalize(self, out_array):
                if self.fill_value is not None:
                    # finish the average. We keep track of the invalid
                    # values (these will have a zero count) set them to
                    # the fill value
                    n = self.count
                    ii = np.isclose(n, np.float32(0.0))
                    n[ii] = np.float32(1.0)
                    tmp = out_array / n
                    tmp[ii] = self.fill_value
                else:
                    tmp = out_array / self.count
                self.count = None
                return tmp

        class summation:
            def __init__(self):
                self.fill_value = None

            def initialize(self, fill_value):
                self.fill_value = fill_value

            def update(self, dev, out_array, in_array):

                # select GPU or CPU
                if dev < 0:
                    np = numpy
                else:
                    np = cupy

                if self.fill_value is not None:
                    # identify the invalid values
                    out_is_bad = np.isclose(out_array, self.fill_value)
                    in_is_bad = np.isclose(in_array, self.fill_value)

                    # accumulate
                    tmp = np.where(out_is_bad, np.float32(0.0), out_array) \
                        + np.where(in_is_bad, np.float32(0.0), in_array)

                else:
                    # accumulate
                    tmp = out_array + in_array

                return tmp

        class minimum:
            def __init__(self):
                self.fill_value = None

            def initialize(self, fill_value):
                self.fill_value = fill_value

            def update(self, out_array, in_array):
                tmp = np.minimum(out_array, in_array)
                # fix invalid values
                if self.fill_value is not None:
                    out_is_bad = np.isclose(out_array, self.fill_value)
                    out_is_good = np.logical_not(out_is_bad)
                    in_is_bad = np.isclose(in_array, self.fill_value)
                    in_is_good = np.logical_not(in_is_bad)
                    tmp = np.where(np.logical_and(out_is_bad, in_is_good), in_array, tmp)
                    tmp = np.where(np.logical_and(in_is_bad, out_is_good), out_array, tmp)
                    tmp = np.where(np.logical_and(in_is_bad, out_is_bad), self.fill_value, tmp)
                return tmp

            def finalize(self, out_array):
                return out_array

        class maximum:
            def __init__(self):
                self.fill_value = None

            def initialize(self, fill_value):
                self.fill_value = fill_value

            def update(self, out_array, in_array):
                tmp = np.maximum(out_array, in_array)
                # fix invalid values
                if self.fill_value is not None:
                    out_is_bad = np.isclose(out_array, self.fill_value)
                    out_is_good = np.logical_not(out_is_bad)
                    in_is_bad = np.isclose(in_array, self.fill_value)
                    in_is_good = np.logical_not(in_is_bad)
                    tmp = np.where(np.logical_and(out_is_bad, in_is_good), in_array, tmp)
                    tmp = np.where(np.logical_and(in_is_bad, out_is_good), out_array, tmp)
                    tmp = np.where(np.logical_and(in_is_bad, out_is_bad), self.fill_value, tmp)
                return tmp

            def finalize(self, out_array):
                return out_array

        def New(self, op_name):
            """ factory method that creates an instance from a string """
            if op_name == 'average':
                return teca_temporal_reduction. \
                    reduction_operator_collection. \
                        average()

            elif op_name == 'minimum':
                return teca_temporal_reduction. \
                    reduction_operator_collection. \
                        minimum()

            elif op_name == 'maximum':
                return teca_temporal_reduction. \
                    reduction_operator_collection. \
                        maximum()

            elif op_name == 'summation':
                return teca_temporal_reduction. \
                    reduction_operator_collection. \
                        summation()


            raise RuntimeError('Invalid operator %s' % (op_name))


    class time_interval:
        """
        Defines a time interval.
        Public member variables:

            time        - the floating point time value of the start of the
                          interval
            start_index - the index into the floating point time array of the
                          first index to include in the interval
            end_index   - the index into the floating point time array of the
                          last index to include in the interval
        """
        def __init__(self, t, start_idx, end_idx, **kwds):
            self.time = t
            self.start_index = start_idx
            self.end_index = end_idx
            self.__dict__.update(kwds)

        def __str__(self):
            strg = ''
            for k, v in self.__dict__.items():
                strg += k + '=' + str(v) + ', '
            return strg

    class interval_iterator_collection:
        """
        A collection of interval_iterators compatible with
        teca_temporal_reduction and a factory method that creates ionstances
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
                    calendar_util.date(t, self.units, self.calendar)

            def __str__(self):
                return '%g (%s, %s) --> %04d-%02d-%02d %02d:%02d:%02g' % (
                    self.t, self.units, self.calendar, self.year, self.month,
                    self.day, self.hour, self.minutes, self.seconds)

        class n_step_iterator:
            """ An iterator over intervals of N time steps """

            def __init__(self, t, units, calendar, n_steps):
                self.time = t
                self.index = 0
                self.n_steps = n_steps

            def __iter__(self):
                return self

            def __next__(self):

                i0 = self.index
                i1 = self.index + self.n_steps

                if i1 >= len(self.time):
                    raise StopIteration

                self.index = i1

                return teca_temporal_reduction. \
                    time_interval(self.time[i0], i0, i1)

        class season_iterator:
            """
            An iterator over seasons (DJF, MAM, JJA, SON) between 2
            time_point's.  A pair of time steps bracketing the current season
            are returned at each iteration. Only full seasonal intervals are
            processed.  If the input data doesn't start or end on a seasonal
            boundary it is skipped.
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
                self.t0 = teca_temporal_reduction. \
                    interval_iterator_collection. \
                        time_point(t[0], units, calendar)

                self.t1 = teca_temporal_reduction. \
                    interval_iterator_collection. \
                        time_point(t[-1], units, calendar)

                # current time state
                self.year, self.month = \
                     self.get_first_season(self.t0.year, self.t0.month)

            def get_season_name(self, month):
                """
                returns one of DJF,MAM,JJA,SON based on the month passed in
                """
                if (month == 12) or ((month >= 1) and (month <= 2)):
                    return 'DJF'
                elif (month >= 3) and (month <= 5):
                    return 'MAM'
                elif (month >= 6) and (month <= 8):
                    return 'JJA'
                elif (month >= 9) and (month <= 11):
                    return 'SON'

                raise RuntimeError('Invalid month %d' % (month))

            def get_first_season(self, y, m):
                """
                given a year and month, checks that the values fall on
                a seasonal boundary. if not, returns the year and month
                of the start of the next season.
                """
                if (m == 12) or (m == 3) or (m == 6) or (m == 9):
                    return y, m
                else:
                    return self.get_next_season(y, m)

            def get_season_end(self, year, month):
                """
                Given a year and month returns the year month and day
                of the end of the season. the input month need not be on
                a seasonal boundary.
                """
                if (month == 12):
                    y = year + 1
                    m = 2
                elif (month >= 1) and (month <= 2):
                    y = year
                    m = 2
                elif (month >= 3) and (month <= 5):
                    y = year
                    m = 5
                elif (month >= 6) and (month <= 8):
                    y = year
                    m = 8
                elif (month >= 9) and (month <= 11):
                    y = year
                    m = 11
                else:
                    raise RuntimeError('Invalid month %d' % (month))

                d = self.last_day_of_month(y, m)

                return y, m, d

            def get_next_season(self, year, month):
                """
                Given a year and month returns the year and month
                of the next season. the input momnth doesn't need to be
                on a seasonal boundary.
                """
                if (month == 12):
                    y = year + 1
                    m = 3
                elif (month >= 1) and (month <= 2):
                    y = year
                    m = 3
                elif (month >= 3) and (month <= 5):
                    y = year
                    m = 6
                elif (month >= 6) and (month <= 8):
                    y = year
                    m = 9
                elif (month >= 9) and (month <= 11):
                    y = year
                    m = 12
                else:
                    raise RuntimeError('Invalid month %d' % (month))

                return y, m

            def last_day_of_month(self, year, month):
                """
                get the number of days in the month, with logic for
                leap years
                """
                return \
                    calendar_util.days_in_month(self.calendar,
                                                self.units, year,
                                                month)

            def __iter__(self):
                return self

            def __next__(self):
                """
                return a pair of time steps bracketing the current month.
                both returned time steps belong to the current month.
                """
                # get the end of the current season
                ey, em, ed = self.get_season_end(self.year, self.month)

                # verify that we have data for the current season
                if ((ey > self.t1.year) or
                    ((ey == self.t1.year) and (em > self.t1.month)) or
                    ((ey == self.t1.year) and (em == self.t1.month) and
                    (ed > self.t1.day))):
                    raise StopIteration

                # find the time step of the first day
                sy = self.year
                sm = self.month

                t0 = '%04d-%02d-01 00:00:00' % (sy, sm)
                i0 = coordinate_util.time_step_of(self.t, False, True,
                                                  self.calendar,
                                                  self.units, t0)

                # find the time step of the last day
                t1 = '%04d-%02d-%02d 23:59:59' % (ey, em, ed)
                i1 = coordinate_util.time_step_of(self.t, True, True,
                                                  self.calendar,
                                                  self.units, t1)

                # move to next season
                self.year, self.month = \
                     self.get_next_season(sy, sm)

                return teca_temporal_reduction.time_interval(
                    self.t[i0], i0, i1, year=sy, month=sm, day=1)


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
                self.t0 = teca_temporal_reduction. \
                    interval_iterator_collection. \
                        time_point(t[0], units, calendar)

                self.t1 = teca_temporal_reduction. \
                    interval_iterator_collection. \
                        time_point(t[-1], units, calendar)

                # current time state
                self.year = self.t0.year
                self.month = self.t0.month

            def last_day_of_month(self):
                """
                get the number of days in the month, with logic for
                leap years
                """
                return \
                    calendar_util.days_in_month(self.calendar,
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
                i0 = coordinate_util.time_step_of(self.t, False, True,
                                                  self.calendar,
                                                  self.units, t0)

                # find the time step of the last day
                n_days = self.last_day_of_month()

                t1 = '%04d-%02d-%02d 23:59:59' % \
                    (self.year, self.month, n_days)

                i1 = coordinate_util.time_step_of(self.t, True, True,
                                                  self.calendar,
                                                  self.units, t1)

                # move to next month
                self.month += 1

                # move to next year
                if self.month == 13:
                    self.month = 1
                    self.year += 1

                return teca_temporal_reduction.time_interval(
                    self.t[i0], i0, i1, year=year, month=month, day=1)

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
                self.t0 = teca_temporal_reduction. \
                    interval_iterator_collection. \
                        time_point(t[0], units, calendar)

                self.t1 = teca_temporal_reduction. \
                    interval_iterator_collection. \
                        time_point(t[-1], units, calendar)

                # current time state
                self.year = self.t0.year
                self.month = self.t0.month
                self.day = self.t0.day

            def last_day_of_month(self):
                """
                get the number of days in the month, with logic for
                leap years
                """
                return calendar_util.days_in_month(
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

                i0 = coordinate_util.time_step_of(self.t, False, True,
                                                  self.calendar,
                                                  self.units, t0)

                # find the time step of the last day
                t1 = '%04d-%02d-%02d 23:59:59' % \
                    (self.year, self.month, self.day)

                i1 = coordinate_util.time_step_of(self.t, True, True,
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

                return teca_temporal_reduction.time_interval(
                    self.t[i0], i0, i1, year=year, month=month, day=day)

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
        self.indices = []
        self.point_arrays = []
        self.interval_name = None
        self.operator_name = None
        self.use_fill_value = 1
        self.fill_value = None
        self.operator = {}

        self.reduction_operator_factory = \
            teca_temporal_reduction.reduction_operator_collection()

        self.interval_iterator_factory = \
            teca_temporal_reduction.interval_iterator_collection()

    def set_reduction_operator_factory(self, factory):
        """
        Sets a factory object that implements a method named New that,
        given a string naming an aoperastor, creates an instance of that
        operator. The factory method has the following signature:

            New(self, op_name) -> reduction_operator

        where:

            op_name - is a string naming the type of reduction operator to
                      create

        """
        self.reduction_operator_factory = factory

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


    def set_fill_value(self, fill_value):
        """
        set the output fill_value
        """
        self.fill_value = fill_value

    def set_use_fill_value(self, use):
        """
        set the output fill_value
        """
        self.use_fill_value = use

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

    def set_point_arrays(self, arrays):
        """
        Set the list of arrays to reduce
        """
        if isinstance(arrays, list):
            arrays = list(arrays)
        self.point_arrays = arrays

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

        # update the metadata so that  modified time axis and reduced variables
        # are presented
        out_atts = teca_metadata()
        out_vars = []

        for array in self.point_arrays:
            # name of the output array
            out_vars.append(array)

            # pass the attributes
            in_atts = atts[array]

            # convert integer to floating point for averaging operations
            if self.operator_name == 'average':
                tc = in_atts['type_code']
                if ( tc == teca_int_array_code.get() )                        \
                        or ( tc == teca_char_array_code.get() )               \
                        or ( tc == teca_short_array_code.get() )              \
                        or ( tc == teca_unsigned_int_array_code.get() )       \
                        or ( tc == teca_unsigned_char_array_code.get() )      \
                        or ( tc == teca_unsigned_short_array_code.get() ):
                    tc = teca_float_array_code.get()
                elif ( tc == teca_long_array_code.get() )                     \
                        or ( tc == teca_long_long_array_code.get() )          \
                        or ( tc == teca_unsigned_long_array_code.get() )      \
                        or ( tc == teca_unsigned_long_long_array_code.get() ):
                    tc = teca_double_array_code.get()
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

        md = md_in[0]

        # initialize a new reduction operator, for the subsequent
        # execute
        atrs = md['attributes']
        for array in self.point_arrays:
            # get the fill value
            fill_value = self.fill_value
            if self.use_fill_value and fill_value is None:
                array_atrs = atrs[array]
                if array_atrs.has('_FillValue'):
                    fill_value = array_atrs['_FillValue']
                elif array_atrs.has('missing_value'):
                    fill_value = array_atrs['missing_value']
                else:
                    raise RuntimeError('Array %s has no fill value. With use_'
                                       'fill_value arrays must have _FillValue'
                                       ' or missing_value attribute or you '
                                       'must set a fill_value explicitly.'%(
                                       array))

            # create and initialize the operator
            op = self.reduction_operator_factory.New(self.operator_name)

            op.initialize(fill_value)

            # save the operator
            self.operator[array] = op

        # generate one request for each time step in the interval
        up_reqs = []

        request_key = md['index_request_key']
        req_id = req_in[request_key]
        ii = self.indices[req_id]
        i = ii.start_index
        while i <= ii.end_index:
            req = teca_metadata(req_in)
            req[request_key] = i
            up_reqs.append(req)
            i += 1

        return up_reqs

    def execute(self, port, data_in, req_in, streaming):
        """
        implements the execute phase of pipeline execution
        """

        # get the requested index
        request_key = req_in['index_request_key']
        req_id = req_in[request_key]
        ii = self.indices[req_id]

        alloc = variant_array_allocator_malloc

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
        mesh_in = as_teca_cartesian_mesh(data_in.pop())
        mesh_out = teca_cartesian_mesh.New()
        mesh_out.copy(mesh_in, alloc)
        arrays_out = mesh_out.get_point_arrays()

        # accumulate incoming values
        while len(data_in):
            mesh_in = as_teca_cartesian_mesh(data_in.pop())
            arrays_in = mesh_in.get_point_arrays()
            for array in self.point_arrays:
                arrays_out[array] = \
                    self.operator[array].update(arrays_out[array],
                                                arrays_in[array])

        # when all the data is processed
        if not streaming:
            # finalize reduction
            for array in self.point_arrays:
                arrays_out[array] = \
                    self.operator[array].finalize(arrays_out[array])

            # fix time
            mesh_out.set_time_step(req_id)
            mesh_out.set_time(ii.time)

        return mesh_out
