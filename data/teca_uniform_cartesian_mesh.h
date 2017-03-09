#ifndef teca_uniform_cartesian_mesh_h
#define teca_uniform_cartesian_mesh_h

#include "teca_uniform_cartesian_mesh_fwd.h"
#include "teca_mesh.h"

/// data on a uniform cartesian mesh
class teca_uniform_cartesian_mesh : public teca_mesh
{
public:
    TECA_DATASET_STATIC_NEW(teca_uniform_cartesian_mesh)
    TECA_DATASET_NEW_INSTANCE()
    TECA_DATASET_NEW_COPY()

    virtual ~teca_uniform_cartesian_mesh() = default;

    // set/get metadata
    TECA_DATASET_METADATA(time, double, 1)
    TECA_DATASET_METADATA(time_step, unsigned long, 1)
    TECA_DATASET_METADATA(spacing, double, 3)
    TECA_DATASET_METADATA(origin, double, 3)
    TECA_DATASET_METADATA(extent, unsigned long, 6)
    TECA_DATASET_METADATA(local_extent, unsigned long, 6)

    // copy data and metadata. shallow copy uses reference
    // counting, while copy duplicates the data.
    void copy(const const_p_teca_dataset &) override;
    void shallow_copy(const p_teca_dataset &) override;

    // swap internals of the two objects
    void swap(p_teca_dataset &) override;

protected:
    teca_uniform_cartesian_mesh();
};

#endif
