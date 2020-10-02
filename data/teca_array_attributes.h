#ifndef teca_array_attributes_h
#define teca_array_attributes_h

#include "teca_metadata.h"
#include <ostream>

// a convenience container for conventional array attributes.
// the attributes listed here are used for CF I/O.
//
// type_code - storage type as defined by teca_variant_array::type_code()
//
// centering - one of: no_centering, point_centering, cell_centering,
//             edge_centering, or face_centering
//
// size - number of elements in the array
//
// units - string describing the uints that the variable is in.
//
// long name - a more descriptive name
//
// description - text describing the data
struct teca_array_attributes
{
    teca_array_attributes() : type_code(0),
        centering(0), size(0), units(), long_name(), description()
    {}

    teca_array_attributes(unsigned int tc, unsigned int cen,
        unsigned long n, const std::string &un, const std::string ln,
        const std::string &descr) : type_code(tc), centering(cen),
        size(n), units(un), long_name(ln), description(descr)
    {}

    teca_array_attributes(const teca_array_attributes &) = default;
    teca_array_attributes &operator=(const teca_array_attributes &) = default;

    // converts from metadata object.
    teca_array_attributes(const teca_metadata &md) :
        type_code(0), centering(0), size(0), units(), long_name(),
        description()
    {
        from(md);
    }

    // convert from metadata object
    teca_array_attributes &operator=(const teca_metadata &md);

    // converts to a metadata object
    operator teca_metadata() const;

    // adds current values to the metadata object
    int to(teca_metadata &md) const;

    // adds current values to the metadata object,
    // only if they don't exist
    int merge_to(teca_metadata &md) const;

    // intializes values from the metadata object
    int from(const teca_metadata &md);

    // send to the stream in human readable form
    void to_stream(std::ostream &os);

    // possible centrings
    //
    // for coordinate system with orthogonal axes x,y,z
    // relative to cell centering:
    //
    //   If A is one of x,y or z then A_face_centering data is located on the
    //   low A face i.e. shifted in the -A direction and arrays will be longer
    //   by 1 value in the A direction.
    //
    //   If A is one of x,y or z then  A_edge_centering data is located on the
    //   low side edge parallel to A corrdinate axis. i.e. shifted in the -B
    //   and -C directions and arrays will be longer by 1 value in the B and C
    //   directions.
    //
    //   point_centering data is located on the low corner. i.e. shifted
    //   in the -A,-B, and -C directions and arrays will be longer
    //   by 1 value in the A, B and C directions.
    //
    // arrays that are not associated with geometric locations should
    // be identified as no_centering.
    //
    // the default centering is cell centering.
    enum
    {
        invalid_value    = 0,
        cell_centering   = 0x0100,
        x_face_centering = 0x0201,
        y_face_centering = 0x0202,
        z_face_centering = 0x0203,
        x_edge_centering = 0x0401,
        y_edge_centering = 0x0402,
        z_edge_centering = 0x0403,
        point_centering  = 0x0800,
        no_centering     = 0x1000,
    };

    unsigned int type_code;
    unsigned int centering;
    unsigned long size;
    std::string units;
    std::string long_name;
    std::string description;
};

#endif
