import teca_py

class teca_python_algorithm(object):
    """
    The base class used for writing new algorithms in Python.
    Contains plumbing that connects user provided callbacks
    to an instance of teca_programmable_algorithm. Users are
    expected to override one or more of get_report_callback,
    get_request_callback, and/or get_execute_callback. These
    methods return a callable with the correct signature, and
    use a closure to access class state.
    """
    @classmethod
    def New(derived_class):
        """ factory method returns an instance of the derived type """
        dc = derived_class()
        dc.initialize()
        return dc

    def initialize(self, n_inputs=1, n_outputs=1):
        """
        Initializes the instance and wires up the plumbing
        """
        self.impl = teca_py.teca_programmable_algorithm.New()
        self.impl.set_number_of_input_connections(n_inputs)
        self.impl.set_number_of_output_ports(n_outputs)
        self.impl.set_name(self.__class__.__name__)
        self.impl.set_report_callback(self.get_report_callback())
        self.impl.set_request_callback(self.get_request_callback())
        self.impl.set_execute_callback(self.get_execute_callback())

    def __getattr__(self, *args):
        """ forward stuff to the programmable algorithm """
        return self.impl.__getattribute__(*args)

    def get_report_callback(self):
        """
        Returns a function with the signature

            report_callback(port, md_in) -> teca_metadata

        The default implementation passes the report down stream
        """
        def report_callback(port, md_in):
             return teca_py.teca_metadata(md_in[0])
        return report_callback

    def get_request_callback(self):
        """
        Returns a function with the signature

            request_callback(port, md_in, req_in) -> [teca_metadata]

        The default implementation passes the request up
        """
        def request_callback(port, md_in, req_in):
            return [teca_py.teca_metadata(req_in)]
        return request_callback

    def get_execute_callback(self):
        """
        Returns a function with the signature

            execute_callback(port, data_in, req_in) -> teca_dataset

        The default implementation shallow copies the input
        """
        def execute_callback(port, data_in, req_in):
            if len(data_in):
                data_out = data_in[0].new_instance()
                data_out.shallow_copy(teca_py.as_non_const_teca_dataset(data_out))
            return data_out
        return execute_callback
