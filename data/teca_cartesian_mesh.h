#ifndef teca_cartesian_mesh_h
#define teca_cartesian_mesh_h

#include "teca_mesh.h"
#include "teca_cartesian_mesh_fwd.h"

/// data on a physically uniform cartesian mesh
class teca_cartesian_mesh : public teca_mesh
{
public:
    TECA_DATASET_STATIC_NEW(teca_cartesian_mesh)
    virtual ~teca_cartesian_mesh() = default;

    // set/get metadata
    TECA_DATASET_METADATA(time, double, 1, m_impl->metadata)
    TECA_DATASET_METADATA(calendar, std::string, 1, m_impl->metadata)
    TECA_DATASET_METADATA(time_units, std::string, 1, m_impl->metadata)
    TECA_DATASET_METADATA(time_step, unsigned long, 1, m_impl->metadata)
    TECA_DATASET_METADATA(whole_extent, unsigned long, 6, m_impl->metadata)
    TECA_DATASET_METADATA(extent, unsigned long, 6, m_impl->metadata)

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

    // virtual constructor. return a new dataset of the same type.
    virtual p_teca_dataset new_instance() const override
    { return teca_cartesian_mesh::New(); }

    // virtual copy constructor. return a deep copy of this
    // dataset in a new instance.
    virtual p_teca_dataset new_copy() const override;

    // copy data and metadata. shallow copy uses reference
    // counting, while copy duplicates the data.
    virtual void copy(const const_p_teca_dataset &) override;
    virtual void shallow_copy(const p_teca_dataset &) override;

    // copy metadata. always a deep copy.
    virtual void copy_metadata(const const_p_teca_dataset &) override;

    // swap internals of the two objects
    virtual void swap(p_teca_dataset &) override;

    // serialize the dataset to/from the given stream
    // for I/O or communication
    virtual void to_stream(teca_binary_stream &) const override;
    virtual void from_stream(teca_binary_stream &) override;

    // stream to/from human readable representation
    virtual void to_stream(std::ostream &) const override;
    virtual void from_stream(std::istream &) override {}

protected:
    teca_cartesian_mesh();

private:
    p_teca_array_collection m_coordinate_arrays;
};

#endif
