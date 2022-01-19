
class teca_python_reduce(object):
    """
    The base class used for writing new reductions in Python.  Contains
    plumbing that connects user provided overrides to an instance of
    teca_programmable_reduce. Users are expected to override one or more of
    report, request, reduce, and/or finalize methods.
    """

    @classmethod
    def New(derived_class):
        """
        factory method returns an instance of the derived type
        """
        dc = derived_class()
        dc.initialize_implementation()
        return dc

    def initialize_implementation(self):
        """
        Initializes the instance and wires up the plumbing.
        """
        # call overrides to get implementation for teca execution
        # phase implementations
        self.impl = teca_programmable_reduce.New()
        self.impl.set_name(self.__class__.__name__)
        self.impl.set_report_callback(self.get_report_callback())
        self.impl.set_request_callback(self.get_request_callback())
        self.impl.set_reduce_callback(self.get_reduce_callback())
        self.impl.set_finalize_callback(self.get_finalize_callback())

    def __getattr__(self, name):
        """
        forward calls to the programmable reduce
        """

        # guard against confusing infinite recursion that
        # occurs if impl is not present. one common way
        # that this occurs is if the instance was not
        # created with the New method
        if name == 'impl':
            raise RuntimeError('The teca_python_reduce ' \
                'was imporperly initialized. Did you use the ' \
                'factory method, New(), to create this ' \
                'instance of %s?'%(self.__class__.__name__))

        # forward to the teca_programmable_reduce
        return self.impl.__getattribute__(name)

    def get_report_callback(self):
        """
        returns a callback to be used by the programmable reduce that
        forwards calls to the class method.
        """
        def report_callback(port, md_in):
            return self.report(port, md_in)
        return report_callback

    def get_request_callback(self):
        """
        returns a callback to be used by the programmable reduce that
        forwards calls to the class method.
        """
        def request_callback(port, md_in, req_in):
            return self.request(port, md_in, req_in)
        return request_callback

    def get_execute_callback(self):
        """
        returns a callback to be used by the programmable reduce that
        forwards calls to the class method.
        """
        def execute_callback(port, data_in, req_in):
            return self.execute(port, data_in, req_in)
        return execute_callback

    def get_reduce_callback(self):
        """
        returns a callback used by the programmable reduce that forwards
        calls to the class method
        """
        def reduce_callback(dev, left, right):
            return self.reduce(dev, left, right)
        return reduce_callback

    def get_finalize_callback(self):
        """
        returns a callback used by the programmable reduce that forwards
        calls to the class method
        """
        def finalize_callback(dev, data):
            return self.finalize(dev, data)
        return finalize_callback

    def report(self, port, md_in):
        """
        return the metadata decribing the data available for consumption.
        Override this to customize the behavior of the report phase of
        execution. The default passes metadata on the first input through.
        """
        return teca_metadata(md_in[0])

    def request(self, port, md_in, req_in):
        """
        return the request for needed data for execution. Override this to
        customize the behavior of the request phase of execution. The default
        passes the request on the first input port through.
        """
        return [teca_metadata(req_in)]

    def reduce(self, dev, left, right):
        """
        given two input datasets return the reduced data. Override this to customize
        the behavior of the reduction. the default raises an exception, this must be
        overridden.
        """
        raise RuntimeError('%s::reduce method was not overridden'%(self.get_class_name()))

    def finalize(self, dev, data_in):
        """
        Called after the reduction is complete. Override this method to customize the
        finalization of the reduction. the default passes the dataset through.
        """
        return data_in
