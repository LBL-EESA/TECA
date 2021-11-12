#ifndef teca_uuid_h
#define teca_uuid_h

#include "teca_config.h"
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>


/// A universally uniquer identifier.
class TECA_EXPORT teca_uuid : public boost::uuids::uuid
{
public:
    teca_uuid() : boost::uuids::uuid(boost::uuids::random_generator()())
    {}

    explicit
    teca_uuid(boost::uuids::uuid const& u) : boost::uuids::uuid(u)
    {}

    operator boost::uuids::uuid() {
        return static_cast<boost::uuids::uuid&>(*this);
    }

    operator boost::uuids::uuid() const {
        return static_cast<boost::uuids::uuid const&>(*this);
    }
};

#endif
