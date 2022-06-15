#ifndef teca_mesh_h
#define teca_mesh_h

#include "teca_config.h"
#include "teca_dataset.h"
#include "teca_metadata.h"
#include "teca_array_collection.h"
#include "teca_shared_object.h"

TECA_SHARED_OBJECT_FORWARD_DECL(teca_mesh)

/// A base class for geometric data.
/**
 * The mesh declares containers for typical geometrically associated data
 * such as point, cell, face and edge centered data arrays and defines
 * the APIs for accessing them. APIs for accessing common metadata such
 * as time related metadata are declared here.
 */
class TECA_EXPORT teca_mesh : public teca_dataset
{
public:
    ~teca_mesh() = default;

    /// @name temporal metadata
    /** Specifies the temporal extents of the data and the calendaring system
     * used to define the time axis.
     */
    ///@{
    TECA_DATASET_METADATA(temporal_bounds, double, 2)
    TECA_DATASET_METADATA(temporal_extent, unsigned long, 2)
    TECA_DATASET_METADATA(calendar, std::string, 1)
    TECA_DATASET_METADATA(time_units, std::string, 1)

    /// Set the temporal_extent to a single time step
    void set_time_step(const unsigned long &val);

    /** Get the time step. note: no checking is done to ensure that the temporal
     * extent has only a single step.
     */
    int get_time_step(unsigned long &val) const;

    /// Set the temporal bounds to a single time value.
    void set_time(const double &val);

    /** Get the time. note: no checking is done to ensure that the temporal
     * extent/bounds have only a single step.
     */
    int get_time(double &val) const;
    ///@}

    /// @name attribute metadata
    /** Provides access to the array attributes metadata, which contains
     * information such as dimensions, units, data type, description, etc.
     */
    ///@{
    TECA_DATASET_METADATA(attributes, teca_metadata, 1)
    ///@}

    /// @name arrays
    /** get the array collection for the given centering the centering
     * enumeration is defined in teca_array_attributes the centering attribute
     * is typically stored in array attribute metadata
     */
    p_teca_array_collection &get_arrays(int centering);
    const_p_teca_array_collection get_arrays(int centering) const;

    /// @name point centered data
    /** returns the array collection for point centered data */
    ///@{
    p_teca_array_collection &get_point_arrays()
    { return m_impl->point_arrays; }

    const_p_teca_array_collection get_point_arrays() const
    { return m_impl->point_arrays; }
    ///@}

    /// @name cell centered data
    /** returns the array collection for edge centered data */
    ///@{
    p_teca_array_collection &get_cell_arrays()
    { return m_impl->cell_arrays; }

    const_p_teca_array_collection get_cell_arrays() const
    { return m_impl->cell_arrays; }
    ///@}

    /// @name edge centered data
    /** returns the array collection for edge centered data */
    ///@{
    p_teca_array_collection &get_x_edge_arrays()
    { return m_impl->x_edge_arrays; }

    const_p_teca_array_collection get_x_edge_arrays() const
    { return m_impl->x_edge_arrays; }

    p_teca_array_collection &get_y_edge_arrays()
    { return m_impl->y_edge_arrays; }

    const_p_teca_array_collection get_y_edge_arrays() const
    { return m_impl->y_edge_arrays; }

    p_teca_array_collection &get_z_edge_arrays()
    { return m_impl->z_edge_arrays; }

    const_p_teca_array_collection get_z_edge_arrays() const
    { return m_impl->z_edge_arrays; }
    ///@}

    /// @name face centered data
    /** returns the array collection for face centered data */
    ///@{
    p_teca_array_collection &get_x_face_arrays()
    { return m_impl->x_face_arrays; }

    const_p_teca_array_collection get_x_face_arrays() const
    { return m_impl->x_face_arrays; }

    p_teca_array_collection &get_y_face_arrays()
    { return m_impl->y_face_arrays; }

    const_p_teca_array_collection get_y_face_arrays() const
    { return m_impl->y_face_arrays; }

    p_teca_array_collection &get_z_face_arrays()
    { return m_impl->z_face_arrays; }

    const_p_teca_array_collection get_z_face_arrays() const
    { return m_impl->z_face_arrays; }
    ///@}

    /// @name non-geometric data
    /** returns the array collection for uncentered data */
    ///@{
    p_teca_array_collection &get_information_arrays()
    { return m_impl->info_arrays; }

    const_p_teca_array_collection get_information_arrays() const
    { return m_impl->info_arrays; }
    ///@}

    /// get the number of points in the mesh
    virtual unsigned long get_number_of_points() const = 0;

    /// get the number of cells in the mesh
    virtual unsigned long get_number_of_cells() const = 0;

    /// return true if the dataset is empty.
    bool empty() const noexcept override;

    /// @copydoc teca_dataset::copy(const const_p_teca_dataset &,allocator)
    void copy(const const_p_teca_dataset &other,
        allocator alloc = allocator::malloc) override;

    /// @copydoc teca_dataset::shallow_copy(const p_teca_dataset &)
    void shallow_copy(const p_teca_dataset &other) override;

    /** append array based data from another mesh. No consistency
     * checks are performed. */
    void append_arrays(const const_p_teca_mesh &);
    void shallow_append_arrays(const p_teca_mesh &);

    /// swap internals of the two objects
    void swap(const p_teca_dataset &) override;

    /** serialize the dataset to/from the given stream
     * for I/O or communication */
    int to_stream(teca_binary_stream &) const override;
    int from_stream(teca_binary_stream &) override;

    /// stream to/from human readable representation
    int to_stream(std::ostream &) const override;
    int from_stream(std::istream &) override { return -1; }

#if defined(SWIG)
protected:
#else
public:
#endif
    /// @note constructors are public to enable std::make_shared. do not use.
    teca_mesh();

public:
    struct impl_t
    {
        impl_t();
        //
        p_teca_array_collection cell_arrays;
        p_teca_array_collection x_edge_arrays;
        p_teca_array_collection y_edge_arrays;
        p_teca_array_collection z_edge_arrays;
        p_teca_array_collection x_face_arrays;
        p_teca_array_collection y_face_arrays;
        p_teca_array_collection z_face_arrays;
        p_teca_array_collection point_arrays;
        p_teca_array_collection info_arrays;
        p_teca_array_collection invalid;
    };
    std::shared_ptr<impl_t> m_impl;
};

#endif
