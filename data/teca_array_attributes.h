#ifndef teca_array_attributes_h
#define teca_array_attributes_h

#include "teca_config.h"
#include "teca_metadata.h"

#include <ostream>
#include <variant>
#include <array>

/** @brief
 * A convenience container for conventional array attributes necessary and/or
 * useful when producing NetCDF CF format files using the teca_cf_writer.
 *
 * @details
 *
 * | Member          | Description                                                |
 * | ------          | -----------                                                |
 * | type_code       | storage type as defined by teca_variant_array::type_code() |
 * | centering       | one of: no_centering, point_centering, cell_centering,     |
 * |                 | edge_centering, or face_centering                          |
 * | size            | number of elements in the array                            |
 * | mesh_dim_active | a 4 tuple x,y,z,t set to 1 if the dimension is used.       |
 * | units           | string describing the units that the variable is in.       |
 * | long name       | a more descriptive name                                    |
 * | description     | text describing the data                                   |
 * | have_fill_value | set non-zero to indicate that a fill_value has been        |
 * |                 | provided.                                                  |
 * | fill_value      | value used to identify missing or invalid data             |
 */
struct TECA_EXPORT teca_array_attributes
{
    teca_array_attributes() : type_code(0),
        centering(0), size(0), mesh_dim_active{},
        units(), long_name(), description(), have_fill_value(0),
        fill_value(1e20f)
    {}

    template <typename fv_t = float>
    teca_array_attributes(unsigned int tc, unsigned int cen,
        unsigned long n, const std::array<int,4> &mda, const std::string &un,
        const std::string &ln, const std::string &descr, const int &have_fv=0,
        const fv_t &fv=fv_t(1e20f)) :
        type_code(tc), centering(cen), size(n), mesh_dim_active(mda), units(un),
        long_name(ln), description(descr), have_fill_value(have_fv), fill_value(fv)
    {}

    teca_array_attributes(const teca_array_attributes &) = default;
    teca_array_attributes &operator=(const teca_array_attributes &) = default;

    /// Convert a from metadata object.
    teca_array_attributes(const teca_metadata &md) :
        type_code(0), centering(0), size(0), mesh_dim_active{{}},
        units(), long_name(), description(), have_fill_value(0),
        fill_value(1.e20f)
    {
        from(md);
    }

    /// Convert from metadata object.
    teca_array_attributes &operator=(const teca_metadata &md);

    /// Converts to a metadata object.
    operator teca_metadata() const;

    /// Adds current values to the metadata object
    int to(teca_metadata &md) const;

    /// Adds the current values to the metadata object, only if they don't exist.
    int merge_to(teca_metadata &md) const;

    /// Intializes values from the metadata object.
    int from(const teca_metadata &md);

    /// Send to the stream in human readable form.
    void to_stream(std::ostream &os) const;

    /** The possible mesh centrings.
     *
     * For coordinate system with orthogonal axes x,y,z relative to cell
     * centering:
     *
     * > If A is one of x,y or z then A_face_centering data is located on the
     * > low A face i.e. shifted in the -A direction and arrays will be longer
     * > by 1 value in the A direction.
     * >
     * > If A is one of x,y or z then  A_edge_centering data is located on the
     * > low side edge parallel to A corrdinate axis. i.e. shifted in the -B
     * > and -C directions and arrays will be longer by 1 value in the B and C
     * > directions.
     * >
     * > point_centering data is located on the low corner. i.e. shifted
     * > in the -A,-B, and -C directions and arrays will be longer
     * > by 1 value in the A, B and C directions.
     *
     * Arrays that are not associated with geometric locations should
     * be identified as no_centering.
     *
     * The default centering is cell centering.
     */
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

    /// convert the centering code to a string
    static const char *centering_to_string(int cen);

    /// flags for non-mesh based arrays
    static constexpr std::array<int,4> none_active() { return {0,0,0,0}; }

    /// flags for time varying 3D arrays
    static constexpr std::array<int,4> xyzt_active() { return {1,1,1,1}; }

    /// flags for time varying 2D arrays
    static constexpr std::array<int,4> xyt_active() { return {1,1,0,1}; }

    /// flags for non-time varying 3D arrays
    static constexpr std::array<int,4> xyz_active() { return {1,1,1,0}; }

    /// flags for non-time varying 2D arrays
    static constexpr std::array<int,4> xy_active() { return {1,1,0,0}; }

    using fill_value_t =
        std::variant<char, unsigned char, short, unsigned short,
            int, unsigned int, long, unsigned long, long long,
            unsigned long long, float, double>;

    unsigned int type_code;
    unsigned int centering;
    unsigned long size;
    std::array<int,4> mesh_dim_active;
    std::string units;
    std::string long_name;
    std::string description;
    int have_fill_value;
    fill_value_t fill_value;
};

#endif
