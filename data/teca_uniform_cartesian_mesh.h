#ifndef teca_uniform_cartesian_mesh_h
#define teca_uniform_cartesian_mesh_h

#include "teca_mesh.h"
#include "teca_shared_object.h"

TECA_SHARED_OBJECT_FORWARD_DECL(teca_uniform_cartesian_mesh)

/// Data on a uniform cartesian mesh.
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

    /// return a unique string identifier
    std::string get_class_name() const override
    { return "teca_uniform_cartesian_mesh"; }

    /// return a unique integer identifier
    int get_type_code() const override;

    /// copy data and metadata. shallow copy uses reference
    /// counting, while copy duplicates the data.
    void copy(const const_p_teca_dataset &) override;
    void shallow_copy(const p_teca_dataset &) override;

    /// swap internals of the two objects
    void swap(p_teca_dataset &) override;

protected:
    teca_uniform_cartesian_mesh();
};

#endif
