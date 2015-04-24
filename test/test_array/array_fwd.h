#ifndef array_fwd_h
#define array_fwd_h

#include <memory>

class array;
using p_array = std::shared_ptr<array>;
using const_p_array = std::shared_ptr<const array>;

class teca_binary_stream;

#endif
