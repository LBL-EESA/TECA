#ifndef teca_uniform_cartesian_mesh_h
#define teca_uniform_cartesian_mesh_h

#include "teca_mesh.h"

class teca_uniform_cartesian_mesh;

using p_teca_uniform_cartesian_mesh
    = std::shared_ptr<teca_uniform_cartesian_mesh>;

using const_p_teca_uniform_cartesian_mesh
    = std::shared_ptr<const teca_uniform_cartesian_mesh>;

/// data on a uniform cartesian mesh
class teca_uniform_cartesian_mesh : public teca_mesh
{
public:
    TECA_DATASET_STATIC_NEW(teca_uniform_cartesian_mesh)
    virtual ~teca_uniform_cartesian_mesh() = default;

    // set/get metadata
    TECA_DATASET_METADATA(time, double, 1, m_impl->metadata)
    TECA_DATASET_METADATA(time_step, unsigned long, 1, m_impl->metadata)
    TECA_DATASET_METADATA(spacing, double, 3, m_impl->metadata)
    TECA_DATASET_METADATA(origin, double, 3, m_impl->metadata)
    TECA_DATASET_METADATA(extent, unsigned long, 6, m_impl->metadata)
    TECA_DATASET_METADATA(local_extent, unsigned long, 6, m_impl->metadata)

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

protected:
    teca_uniform_cartesian_mesh();
};

#endif
