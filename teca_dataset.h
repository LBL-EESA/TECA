#ifndef teca_dataset_h
#define teca_dataset_h

#include "teca_dataset_fwd.h"
#include "teca_compiler.h"
class teca_binary_stream;

class teca_dataset : public std::enable_shared_from_this<teca_dataset>
{
public:
    virtual ~teca_dataset(){}

    // covert to bool. true if the dataset is not empty.
    // otherwise false.
    explicit operator bool() TECA_NOEXCEPT
    { return !this->empty(); }

    // return true if the dataset is empty.
    virtual bool empty() TECA_NOEXCEPT
    { return true; }

    // return a new dataset of the same type
    virtual p_teca_dataset new_instance()
    { return p_teca_dataset(new teca_dataset); }

    // serialize the dataset to/from the given stream
    // for I/O or communication
    virtual void to_stream(teca_binary_stream &) {}
    virtual void from_stream(teca_binary_stream &) {}

// TODO

protected:
    teca_dataset(){}
};

#endif
