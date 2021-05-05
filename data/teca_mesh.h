#ifndef teca_mesh_h
#define teca_mesh_h

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
class teca_mesh : public teca_dataset
{
public:
    ~teca_mesh() = default;

    // set/get temporal metadata
    TECA_DATASET_METADATA(time, double, 1)
    TECA_DATASET_METADATA(calendar, std::string, 1)
    TECA_DATASET_METADATA(time_units, std::string, 1)
    TECA_DATASET_METADATA(time_step, unsigned long, 1)

    // set/get attribute metadata
    TECA_DATASET_METADATA(attributes, teca_metadata, 1)

    // get the array collection for the given centering
    // the centering enumeration is defined in teca_array_attributes
    // the centering attribute is typically stored in array attribute
    // metadata
    p_teca_array_collection &get_arrays(int centering);
    const_p_teca_array_collection get_arrays(int centering) const;

    // get point centered data
    p_teca_array_collection &get_point_arrays()
    { return m_impl->point_arrays; }

    const_p_teca_array_collection get_point_arrays() const
    { return m_impl->point_arrays; }

    // get cell centered data
    p_teca_array_collection &get_cell_arrays()
    { return m_impl->cell_arrays; }

    const_p_teca_array_collection get_cell_arrays() const
    { return m_impl->cell_arrays; }

    // get edge centered data
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

    // get face centered data
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

    // get non-geometric data
    p_teca_array_collection &get_information_arrays()
    { return m_impl->info_arrays; }

    const_p_teca_array_collection get_information_arrays() const
    { return m_impl->info_arrays; }

    // return true if the dataset is empty.
    bool empty() const noexcept override;

    // copy data and metadata. shallow copy uses reference
    // counting, while copy duplicates the data.
    void copy(const const_p_teca_dataset &) override;
    void shallow_copy(const p_teca_dataset &) override;

    // append array based data from another mesh. No consistency
    // checks are performed.
    void append_arrays(const const_p_teca_mesh &);
    void shallow_append_arrays(const p_teca_mesh &);

    // swap internals of the two objects
    void swap(p_teca_dataset &) override;

    // serialize the dataset to/from the given stream
    // for I/O or communication
    int to_stream(teca_binary_stream &) const override;
    int from_stream(teca_binary_stream &) override;

    // stream to/from human readable representation
    int to_stream(std::ostream &) const override;
    int from_stream(std::istream &) override { return -1; }

protected:
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
