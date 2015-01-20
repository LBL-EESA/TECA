#ifndef teca_dataset_h
#define teca_dataset_h

#include "teca_dataset_fwd.h"
class teca_binary_stream;

class teca_dataset : public std::enable_shared_from_this<teca_dataset>
{
public:
    virtual ~teca_dataset(){}
    explicit operator bool()
    { return !this->empty(); }

    virtual bool empty(){ return true; }

    virtual p_teca_dataset new_instance()
    { return p_teca_dataset(new teca_dataset); }

    virtual void to_stream(teca_binary_stream &s){}
    virtual void from_stream(teca_binary_stream &s){}

// TODO

protected:
    teca_dataset(){}
};

#endif
