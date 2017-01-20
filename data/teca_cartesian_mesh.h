#ifndef teca_cartesian_mesh_h
#define teca_cartesian_mesh_h

#include "teca_mesh.h"
#include "teca_cartesian_mesh_fwd.h"

/// data on a physically uniform cartesian mesh
class teca_cartesian_mesh : public teca_mesh
{
public:
    TECA_DATASET_STATIC_NEW(teca_cartesian_mesh)
    TECA_DATASET_NEW_INSTANCE()
    TECA_DATASET_NEW_COPY()

    virtual ~teca_cartesian_mesh() = default;

    // set/get metadata
    TECA_DATASET_METADATA(time, double, 1)
    TECA_DATASET_METADATA(calendar, std::string, 1)
    TECA_DATASET_METADATA(time_units, std::string, 1)
    TECA_DATASET_METADATA(time_step, unsigned long, 1)
    TECA_DATASET_METADATA(whole_extent, unsigned long, 6)
    TECA_DATASET_METADATA(extent, unsigned long, 6)

    // get x coordinate array
    p_teca_variant_array get_x_coordinates()
    { return m_coordinate_arrays->get("x"); }

    const_p_teca_variant_array get_x_coordinates() const
    { return m_coordinate_arrays->get("x"); }

    // get y coordinate array
    p_teca_variant_array get_y_coordinates()
    { return m_coordinate_arrays->get("y"); }

    const_p_teca_variant_array get_y_coordinates() const
    { return m_coordinate_arrays->get("y"); }

    // get z coordinate array
    p_teca_variant_array get_z_coordinates()
    { return m_coordinate_arrays->get("z"); }

    const_p_teca_variant_array get_z_coordinates() const
    { return m_coordinate_arrays->get("z"); }

    // set coordinate arrays
    void set_x_coordinates(const p_teca_variant_array &a)
    { m_coordinate_arrays->set("x", a); }

    void set_y_coordinates(const p_teca_variant_array &a)
    { m_coordinate_arrays->set("y", a); }

    void set_z_coordinates(const p_teca_variant_array &a)
    { m_coordinate_arrays->set("z", a); }

    // copy data and metadata. shallow copy uses reference
    // counting, while copy duplicates the data.
    void copy(const const_p_teca_dataset &) override;
    void shallow_copy(const p_teca_dataset &) override;

    // copy metadata. always a deep copy.
    void copy_metadata(const const_p_teca_dataset &other) override;

    // swap internals of the two objects
    void swap(p_teca_dataset &) override;

    // serialize the dataset to/from the given stream
    // for I/O or communication
    void to_stream(teca_binary_stream &) const override;
    void from_stream(teca_binary_stream &) override;

    // stream to/from human readable representation
    void to_stream(std::ostream &) const override;

protected:
    teca_cartesian_mesh();

private:
    p_teca_array_collection m_coordinate_arrays;
};

#endif
