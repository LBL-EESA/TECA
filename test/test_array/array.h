#ifndef array_h
#define array_h

#include "array.h"
#include "teca_dataset.h"
#include "teca_shared_object.h"

#include <hamm_memory_resource.h>
#include <hamm_pmr_vector.h>

#include <string>
#include <sstream>

TECA_SHARED_OBJECT_FORWARD_DECL(array)

class teca_binary_stream;

// trivial implementation of an array based datatset
// for testing the pipeline
class array : public teca_dataset
{
public:
    ~array() {}

    /// create an array that accessible on the CPU
    static p_array new_cpu_accessible();

    /// create an array that accessible from CUDA
    static p_array new_cuda_accessible();

    /// create an array dataset with a specific memory memory_resource.
    static p_array New(const p_hamm_memory_resource &alloc);

    TECA_DATASET_STATIC_NEW(array);

    TECA_DATASET_PROPERTY(std::vector<size_t>, extent)
    TECA_DATASET_PROPERTY(std::string, name)

    // set the contained data
    void set_data(const hamm_pmr_vector<double> &a_data) { *this->data = a_data; }

    // get the contained data
    hamm_pmr_vector<double> &get_data() { return *this->data; }
    const hamm_pmr_vector<double> &get_data() const { return *this->data; }

    // get a pointer to the contained data
    double *get() { return this->data->data(); }
    const double *get() const { return this->data->data(); }

    // check if the data is accessible from CUDA
    bool cuda_accessible() const;

    // check if the data is accessible from the CPU
    bool cpu_accessible() const;

    // return a unique string identifier
    std::string get_class_name() const override
    { return "array"; }

    // return an integer identifier uniquely naming the dataset type
    int get_type_code() const override
    { return 1; }

    // return true if the dataset is empty.
    bool empty() const noexcept override
    { return this->data->empty(); }

    // return a new dataset of the same type
    p_teca_dataset new_instance() const override
    { return p_teca_dataset(new array()); }

    // return a new copy constructed array
    p_teca_dataset new_copy() const override;
    p_teca_dataset new_shallow_copy() override;

    size_t size() const
    { return this->data->size(); }

    void resize(size_t n);
    void clear();

    double &get(size_t i)
    { return this->data->at(i); }

    const double &get(size_t i) const
    { return this->data->at(i); }

    double &operator[](size_t i)
    { return this->data->at(i); }

    const double &operator[](size_t i) const
    { return this->data->at(i); }

    void append(double d)
    { this->data->push_back(d); }

    void copy(const const_p_teca_dataset &) override;
    void shallow_copy(const p_teca_dataset &) override;
    void copy_metadata(const const_p_teca_dataset &) override;
    void swap(const p_teca_dataset &) override;

    // serialize the dataset to/from the given stream
    // for I/O or communication
    int to_stream(teca_binary_stream &s) const override;
    int from_stream(teca_binary_stream &s) override;

    int to_stream(std::ostream &s) const override;
    int from_stream(std::istream &) override { return -1; }

protected:
    /// creates a new array dataset with the default memory resource
    array();

    /// creates a new array dataset with a specific memory resource
    array(const p_hamm_memory_resource &alloc);

    array(const array &);
    void operator=(const array &);

private:
    std::string name;
    std::vector<size_t> extent;

    p_hamm_memory_resource memory_resource;
    std::shared_ptr<hamm_pmr_vector<double>> data;
};

#endif
