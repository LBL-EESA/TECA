#ifndef teca_uniform_cartesian_mesh_h
#define teca_uniform_cartesian_mesh_h

#include "teca_dataset.h"
#include "teca_metadata.h"
#include "teca_array_collection.h"

class teca_uniform_cartesian_mesh;

using p_teca_uniform_cartesian_mesh
    = std::shared_ptr<teca_uniform_cartesian_mesh>;

using const_p_teca_uniform_cartesian_mesh
    = std::shared_ptr<const teca_uniform_cartesian_mesh>;

/// data on a physically uniform cartesian mesh
class teca_uniform_cartesian_mesh : public teca_dataset
{
public:
    TECA_DATASET_STATIC_NEW(teca_uniform_cartesian_mesh)
    virtual ~teca_uniform_cartesian_mesh() = default;

    // set/get metadata
    TECA_DATASET_METADATA(spacing, double, 3, m_impl->metadata)
    TECA_DATASET_METADATA(origin, double, 3, m_impl->metadata)
    TECA_DATASET_METADATA(extent, unsigned long, 6, m_impl->metadata)
    TECA_DATASET_METADATA(local_extent, unsigned long, 6, m_impl->metadata)

    // get point centered data
    p_teca_array_collection get_point_arrays()
    { return m_impl->point_arrays; }

    // get cell centered data
    p_teca_array_collection get_cell_arrays()
    { return m_impl->cell_arrays; }

    // get edge centered data
    p_teca_array_collection get_edge_arrays()
    { return m_impl->edge_arrays; }

    // get face centered data
    p_teca_array_collection get_face_arrays()
    { return m_impl->face_arrays; }

    // return true if the dataset is empty.
    virtual bool empty() const TECA_NOEXCEPT override;

    // virtual constructor. return a new dataset of the same type.
    virtual p_teca_dataset new_instance() const override
    { return teca_uniform_cartesian_mesh::New(); }

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
    virtual void from_stream(std::istream &) {}

protected:
    teca_uniform_cartesian_mesh();

private:
    struct impl_t
    {
        teca_metadata metadata;
        p_teca_array_collection point_arrays;
        p_teca_array_collection cell_arrays;
        p_teca_array_collection edge_arrays;
        p_teca_array_collection face_arrays;
    };
    std::shared_ptr<impl_t> m_impl;
};

#endif
