#ifndef array_source_h
#define array_source_h

#include <memory>
#include <vector>
#include <string>
#include "teca_algorithm.h"

class array_source;
typedef std::shared_ptr<array_source> p_array_source;

/** an example implementation of a teca_algorithm
that generates arrays over a number of timesteps

metadata keys:
     time
     array_names
     array_size

request keys:
     time (required)
     array_name (required)
     extent (required)
*/
class array_source : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(array_source)
    ~array_source();

    // set the number of arrays that coulod be generated.
    // these can be requested names names "array_0" ... "array_n-1"
    void set_number_of_arrays(unsigned int n);
    int get_number_of_arrays() const { return this->array_names.size(); }

    // set the size of the arrays to generate
    TECA_ALGORITHM_PROPERTY(unsigned int, array_size)

    // set the number of timesteps to generate
    TECA_ALGORITHM_PROPERTY(unsigned int, number_of_timesteps)

protected:
    array_source();

private:
    virtual
    teca_meta_data get_output_meta_data(
        unsigned int port,
        std::vector<teca_meta_data> &input_md);

    virtual
    p_teca_dataset execute(
        unsigned int port,
        std::vector<p_teca_dataset> &input_data,
        teca_meta_data &request);

private:
    std::vector<std::string> array_names;
    unsigned int array_size;
    unsigned int number_of_timesteps;
};

#endif
