#ifndef teca_algorithm_output_port_h
#define teca_algorithm_output_port_h

/// @file

/// An output port packages an algorithm and a port number
using teca_algorithm_output_port
    = std::pair<p_teca_algorithm, unsigned int>;

/// get the algorithm from the output port
inline
p_teca_algorithm &get_algorithm(teca_algorithm_output_port &op)
{ return op.first; }

/// get port number from the output port
inline
unsigned int &get_port(teca_algorithm_output_port &op)
{ return op.second; }

#endif
