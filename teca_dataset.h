#ifndef teca_dataset_h
#define teca_dataset_h

#include "teca_dataset_fwd.h"

class teca_dataset : public std::enable_shared_from_this<teca_dataset>
{
public:
    virtual ~teca_dataset(){}
// TODO
protected:
    teca_dataset(){}
};

#endif
