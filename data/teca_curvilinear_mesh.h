#ifndef teca_curvilinear_mesh_h
#define teca_curvilinear_mesh_h

#include "teca_mesh.h"
#include "teca_shared_object.h"
TECA_SHARED_OBJECT_FORWARD_DECL(teca_curvilinear_mesh)

/// Data on a physically uniform curvilinear mesh.
class teca_curvilinear_mesh : public teca_mesh
{
public:
    TECA_DATASET_STATIC_NEW(teca_curvilinear_mesh)
    TECA_DATASET_NEW_INSTANCE()
    TECA_DATASET_NEW_COPY()

    virtual ~teca_curvilinear_mesh() = default;

    // return a unique string identifier
    std::string get_class_name() const override
    { return "teca_curvilinear_mesh"; }

    // return the type code
    int get_type_code() const override;

    // set/get metadata
    TECA_DATASET_METADATA(whole_extent, unsigned long, 6)
    TECA_DATASET_METADATA(extent, unsigned long, 6)
    TECA_DATASET_METADATA(bounds, double, 6)
    TECA_DATASET_METADATA(periodic_in_x, int, 1)
    TECA_DATASET_METADATA(periodic_in_y, int, 1)
    TECA_DATASET_METADATA(periodic_in_z, int, 1)
    TECA_DATASET_METADATA(x_coordinate_variable, std::string, 1)
    TECA_DATASET_METADATA(y_coordinate_variable, std::string, 1)
    TECA_DATASET_METADATA(z_coordinate_variable, std::string, 1)
    TECA_DATASET_METADATA(t_coordinate_variable, std::string, 1)

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
    void set_x_coordinates(const std::string &name,
        const p_teca_variant_array &array);

    void set_y_coordinates(const std::string &name,
        const p_teca_variant_array &array);

    void set_z_coordinates(const std::string &name,
        const p_teca_variant_array &array);

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
    int to_stream(teca_binary_stream &) const override;
    int from_stream(teca_binary_stream &) override;

    // stream to/from human readable representation
    int to_stream(std::ostream &) const override;

protected:
    teca_curvilinear_mesh();

private:
    p_teca_array_collection m_coordinate_arrays;
};

#endif
