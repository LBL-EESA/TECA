#ifndef array_executive_h
#define array_executive_h

#include "teca_shared_object.h"
#include "teca_algorithm_executive.h"
#include "teca_metadata.h"

#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(array_executive)

class array_executive : public teca_algorithm_executive
{
public:
    TECA_ALGORITHM_EXECUTIVE_STATIC_NEW(array_executive)

    virtual int initialize(const teca_metadata &md);
    virtual teca_metadata get_next_request();

protected:
    array_executive(){}

private:
    std::vector<teca_metadata> requests;
};

#endif
