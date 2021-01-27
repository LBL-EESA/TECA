
class teca_python_algorithm(object):
    """
    The base class used for writing new algorithms in Python.  Contains
    plumbing that connects user provided overrides to an instance of
    teca_programmable_algorithm. Users are expected to override one or more of
    report, request, and/or execute.
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
        # call overridable methods to get number of inputs/outputs
        n_inputs = self.get_number_of_input_connections()
        n_outputs = self.get_number_of_output_ports()

        # call overrides to get implementation for teca execution
        # phase implementations
        self.impl = teca_programmable_algorithm.New()
        self.impl.set_number_of_input_connections(n_inputs)
        self.impl.set_number_of_output_ports(n_outputs)
        self.impl.set_name(self.__class__.__name__)
        self.impl.set_report_callback(self.get_report_callback())
        self.impl.set_request_callback(self.get_request_callback())
        self.impl.set_execute_callback(self.get_execute_callback())

    def __getattr__(self, name):
        """
        forward calls to the programmable algorithm
        """

        # guard against confusing infinite recursion that
        # occurs if impl is not present. one common way
        # that this occurs is if the instance was not
        # created with the New method
        if name == 'impl':
            raise RuntimeError('The teca_python_algorithm ' \
                'was imporperly initialized. Did you use the ' \
                'factory method, New(), to create this ' \
                'instance of %s?'%(self.__class__.__name__))

        # forward to the teca_programmable_algorithm
        return self.impl.__getattribute__(name)

    def get_report_callback(self):
        """
        returns a callback to be used by the programmable algorithm that
        forwards calls to the class method.
        """
        def report_callback(port, md_in):
            return self.report(port, md_in)
        return report_callback

    def get_request_callback(self):
        """
        returns a callback to be used by the programmable algorithm that
        forwards calls to the class method.
        """
        def request_callback(port, md_in, req_in):
            return self.request(port, md_in, req_in)
        return request_callback

    def get_execute_callback(self):
        """
        returns a callback to be used by the programmable algorithm that
        forwards calls to the class method.
        """
        def execute_callback(port, data_in, req_in):
            return self.execute(port, data_in, req_in)
        return execute_callback

    def get_number_of_input_connections(self):
        """
        return the number of input connections this algorithm needs.
        The default is 1, override to modify.
        """
        return 1

    def get_number_of_output_ports(self):
        """
        return the number of output ports this algorithm provides.
        The default is 1, override to modify.
        """
        return 1

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

    def execute(self, port, data_in, req_in):
        """
        return the processed data. Override this to customize the behavior of
        the execute phase of execution. The default passes the dataset on the
        first input port through.
        """
        if len(data_in):
            data_out = data_in[0].new_instance()
            data_out.shallow_copy(as_non_const_teca_dataset(data_out))
        return data_out
