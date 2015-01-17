#ifndef teca_algorithm_output_port_h
#define teca_algorithm_output_port_h

typedef std::pair<p_teca_algorithm, unsigned int>
    teca_algorithm_output_port;

// convenience functions for accessing port and algorithm
// from an output port
inline
p_teca_algorithm &get_algorithm(teca_algorithm_output_port &op)
{ return op.first; }

inline
unsigned int &get_port(teca_algorithm_output_port &op)
{ return op.second; }

#endif
