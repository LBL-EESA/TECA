#ifndef teca_arakawa_c_grid_h
#define teca_arakawa_c_grid_h

#include "teca_config.h"
#include "teca_mesh.h"
#include "teca_shared_object.h"
#include "teca_variant_array.h"
#include "teca_array_collection.h"

#include <map>
#include <string>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_arakawa_c_grid)

/// A representation of mesh based data on an Arkawa C Grid.
/**
 * The Arakawa C grid is defined by various combinations of horizontal and
 * vertical centerings.
 *
 * The horizontal centerings occur at so called mass or M points, U points,
 * and V points. These centerings are depicted in the following diagram:
 *
 * >  *-------V-------*
 * >  |               |
 * >  |               |
 * >  |               |
 * >  U       M       U
 * >  |               |
 * >  |               |
 * >  |               |
 * >  *-------V-------*
 *
 * The horizontal coordinates are stored in 2d arrays. Assuming the mass
 * coordinate arrays have dimension [nx, ny], then the U coordinate arrays
 * have dimension [nx + 1, ny], and the V coordinate arrays have dimension
 * [nx, ny + 1].
 *
 * The vertical centerings occur at so called M points and W points. These
 * centerings are depicted in the following diagram.
 *
 * >  *-------W-------*
 * >  |               |
 * >  |               |
 * >  |               |
 * >  |       M       |
 * >  |               |
 * >  |               |
 * >  |               |
 * >  *-------W-------*
 *
 * The vertical coordinates are stored in 1d arrays. Assuming the M vertical
 * coordinate has the dimension [nz], then the W coordinate has dimension
 * [nz + 1].
 *
 * The 3d mesh dimensions can be obtained from mesh metadata, as well as
 * coordinate array names, and array attributes describing the data type,
 * units, etc.
 *
 * Variables may exist on one of a number of permutations of horizontal and
 * vertical centerings, array attributes contains the centering metadata.
 *
 * See also:
 * "A Description of the Advanced Research WRF Model Version 4",
 * NCAR/TN-556+STR
 *
 * "Grids in Numerical Weather and Climate Models"
 * http://dx.doi.org/10.5772/55922
 */
class TECA_EXPORT teca_arakawa_c_grid : public teca_mesh
{
public:
    TECA_DATASET_STATIC_NEW(teca_arakawa_c_grid)
    TECA_DATASET_NEW_INSTANCE()
    TECA_DATASET_NEW_COPY()

    virtual ~teca_arakawa_c_grid() = default;

    // set/get metadata
    TECA_DATASET_METADATA(whole_extent, unsigned long, 6)
    TECA_DATASET_METADATA(extent, unsigned long, 6)
    TECA_DATASET_METADATA(bounds, double, 6)

    // flag set if the boundary in the given direction is periodic
    TECA_DATASET_METADATA(periodic_in_x, int, 1)
    TECA_DATASET_METADATA(periodic_in_y, int, 1)
    TECA_DATASET_METADATA(periodic_in_z, int, 1)

    /// get the number of points in the mesh
    unsigned long get_number_of_points() const override;

    /// get the number of cells in the mesh
    unsigned long get_number_of_cells() const override;

    // get the names of the m, and v horizontal coordinate arrays
    // these should not need to be modified
    TECA_DATASET_METADATA(m_x_coordinate_variable, std::string, 1)
    TECA_DATASET_METADATA(m_y_coordinate_variable, std::string, 1)
    TECA_DATASET_METADATA(u_x_coordinate_variable, std::string, 1)
    TECA_DATASET_METADATA(u_y_coordinate_variable, std::string, 1)
    TECA_DATASET_METADATA(v_x_coordinate_variable, std::string, 1)
    TECA_DATASET_METADATA(v_y_coordinate_variable, std::string, 1)

    // get the names of the m and w vertical coordinate arrays
    // these should not need to be modified
    TECA_DATASET_METADATA(m_z_coordinate_variable, std::string, 1)
    TECA_DATASET_METADATA(w_z_coordinate_variable, std::string, 1)

    // get the name of the time axis arrays
    // this should not need to be modified
    TECA_DATASET_METADATA(t_coordinate_variable, std::string, 1)

    // get m,u, and v horizontal coordinate arrays
    p_teca_variant_array get_m_x_coordinates();
    p_teca_variant_array get_m_y_coordinates();
    const_p_teca_variant_array get_m_x_coordinates() const;
    const_p_teca_variant_array get_m_y_coordinates() const;

    p_teca_variant_array get_u_x_coordinates();
    p_teca_variant_array get_u_y_coordinates();
    const_p_teca_variant_array get_u_x_coordinates() const;
    const_p_teca_variant_array get_u_y_coordinates() const;

    p_teca_variant_array get_v_x_coordinates();
    p_teca_variant_array get_v_y_coordinates();
    const_p_teca_variant_array get_v_x_coordinates() const;
    const_p_teca_variant_array get_v_y_coordinates() const;

    // get m and w z coordinate array
    p_teca_variant_array get_m_z_coordinates();
    const_p_teca_variant_array get_m_z_coordinates() const;

    p_teca_variant_array get_w_z_coordinates();
    const_p_teca_variant_array get_w_z_coordinates() const;

    // get t coordinate array
    p_teca_variant_array get_t_coordinates();
    const_p_teca_variant_array get_t_coordinates() const;

    // set m,u, and v horizontal coordinate arrays
    void set_m_x_coordinates(const std::string &name,
        const p_teca_variant_array &a);

    void set_m_y_coordinates(const std::string &name,
        const p_teca_variant_array &a);

    void set_u_x_coordinates(const std::string &name,
        const p_teca_variant_array &a);

    void set_u_y_coordinates(const std::string &name,
        const p_teca_variant_array &a);

    void set_v_x_coordinates(const std::string &name,
        const p_teca_variant_array &a);

    void set_v_y_coordinates(const std::string &name,
        const p_teca_variant_array &a);

    // get m and w z coordinate array
    void set_m_z_coordinates(const std::string &name,
        const p_teca_variant_array &a);

    void set_w_z_coordinates(const std::string &name,
        const p_teca_variant_array &a);

    // set t coordinate array
    void set_t_coordinates(const std::string &name,
        const p_teca_variant_array &a);

    // return a unique string identifier
    std::string get_class_name() const override
    { return "teca_arakawa_c_grid"; }

    // return the type code
    int get_type_code() const override;

    /// @copydoc teca_dataset::copy(const const_p_teca_dataset &,allocator)
    void copy(const const_p_teca_dataset &other,
        allocator alloc = allocator::malloc) override;

    /// @copydoc teca_dataset::shallow_copy(const p_teca_dataset &)
    void shallow_copy(const p_teca_dataset &other) override;

    // copy metadata. always a deep copy.
    void copy_metadata(const const_p_teca_dataset &other) override;

    // swap internals of the two objects
    void swap(const p_teca_dataset &) override;

    // return true if the dataset is empty.
    bool empty() const noexcept override;

    // serialize the dataset to/from the given stream
    // for I/O or communication
    int to_stream(teca_binary_stream &) const override;
    int from_stream(teca_binary_stream &) override;

    // stream to/from human readable representation
    int to_stream(std::ostream &) const override;
    using teca_dataset::from_stream;

#if defined(SWIG)
protected:
#else
public:
#endif
    // NOTE: constructors are public to enable std::make_shared. do not use.
    teca_arakawa_c_grid();

private:
    struct impl_t
    {
        impl_t();

        p_teca_variant_array m_x_coordinates;
        p_teca_variant_array m_y_coordinates;
        p_teca_variant_array u_x_coordinates;
        p_teca_variant_array u_y_coordinates;
        p_teca_variant_array v_x_coordinates;
        p_teca_variant_array v_y_coordinates;
        p_teca_variant_array m_z_coordinates;
        p_teca_variant_array w_z_coordinates;
        p_teca_variant_array t_coordinates;
    };
    std::shared_ptr<impl_t> m_impl;
};

#endif
