#ifndef teca_uniform_cartesian_mesh_h
#define teca_uniform_cartesian_mesh_h

#include "teca_config.h"
#include "teca_mesh.h"
#include "teca_shared_object.h"

TECA_SHARED_OBJECT_FORWARD_DECL(teca_uniform_cartesian_mesh)

/// Data on a uniform cartesian mesh.
class TECA_EXPORT teca_uniform_cartesian_mesh : public teca_mesh
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
    TECA_DATASET_METADATA(whole_extent, unsigned long, 6)

    /// get the number of points in the mesh
    unsigned long get_number_of_points() const override;

    /// get the number of cells in the mesh
    unsigned long get_number_of_cells() const override;

    /// return a unique string identifier
    std::string get_class_name() const override
    { return "teca_uniform_cartesian_mesh"; }

    /// return a unique integer identifier
    int get_type_code() const override;

    /// @copydoc teca_dataset::copy(const const_p_teca_dataset &,allocator)
    void copy(const const_p_teca_dataset &other,
        allocator alloc = allocator::malloc) override;

    /// @copydoc teca_dataset::shallow_copy(const p_teca_dataset &)
    void shallow_copy(const p_teca_dataset &other) override;

    /// swap internals of the two objects
    void swap(const p_teca_dataset &) override;

#if defined(SWIG)
protected:
#else
public:
#endif
    // NOTE: constructors are public to enable std::make_shared. do not use.
    teca_uniform_cartesian_mesh();
};

#endif
