#ifndef teca_cartesian_mesh_h
#define teca_cartesian_mesh_h

#include "teca_mesh.h"
#include "teca_shared_object.h"

TECA_SHARED_OBJECT_FORWARD_DECL(teca_cartesian_mesh)

/// An object representing data on a stretched Cartesian mesh.
class teca_cartesian_mesh : public teca_mesh
{
public:
    TECA_DATASET_STATIC_NEW(teca_cartesian_mesh)
    TECA_DATASET_NEW_INSTANCE()
    TECA_DATASET_NEW_COPY()

    virtual ~teca_cartesian_mesh() = default;

    // Set/get metadata
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

    /** Get the extent of the named array, taking into account the variable's
     * dimensions as opposed to the mesh's dimensions.  For instance the mesh
     * extent may represent a volume while a variables extent may represent a
     * slice. returns 0 if successful, -1 if an error occurred, 1 if the
     * have_mesh_dims flag is missing. The latter is not necessarily an error.
     */
    int get_array_extent(const std::string &array_name,
        unsigned long array_extent[6]);

    /// Get the x coordinate array
    p_teca_variant_array get_x_coordinates()
    { return m_coordinate_arrays->get("x"); }

    const_p_teca_variant_array get_x_coordinates() const
    { return m_coordinate_arrays->get("x"); }

    /// Get the y coordinate array
    p_teca_variant_array get_y_coordinates()
    { return m_coordinate_arrays->get("y"); }

    const_p_teca_variant_array get_y_coordinates() const
    { return m_coordinate_arrays->get("y"); }

    /// Get the z coordinate array
    p_teca_variant_array get_z_coordinates()
    { return m_coordinate_arrays->get("z"); }

    const_p_teca_variant_array get_z_coordinates() const
    { return m_coordinate_arrays->get("z"); }

    /// Set the x coordinate array
    void set_x_coordinates(const std::string &name,
        const p_teca_variant_array &array);

    /// Set the y coordinate array
    void set_y_coordinates(const std::string &name,
        const p_teca_variant_array &array);

    /// Set the z coordinate array
    void set_z_coordinates(const std::string &name,
        const p_teca_variant_array &array);

    /// Return the name of the class
    std::string get_class_name() const override
    { return "teca_cartesian_mesh"; }

    /// return an integer identifier uniquely naming the dataset type
    int get_type_code() const override;

    /** Copy data and metadata. Shallow copy uses reference
     * counting, while copy duplicates the data.
     */
    void copy(const const_p_teca_dataset &) override;
    void shallow_copy(const p_teca_dataset &) override;

    /// Copy metadata. This is always a deep copy.
    void copy_metadata(const const_p_teca_dataset &other) override;

    /// Swap the internals of the two objects
    void swap(p_teca_dataset &) override;

    /** Serialize the dataset to/from the given stream
     * for I/O or communication
     */
    int to_stream(teca_binary_stream &) const override;
    int from_stream(teca_binary_stream &) override;

    // stream to/from human readable representation
    int to_stream(std::ostream &) const override;

protected:
    teca_cartesian_mesh();

private:
    p_teca_array_collection m_coordinate_arrays;
};

#endif
