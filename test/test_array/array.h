#ifndef array_h
#define array_h

#include "teca_config.h"
#include "teca_dataset.h"
#include "teca_shared_object.h"

#include <hamr_buffer_allocator.h>

#include <string>
#include <sstream>

TECA_SHARED_OBJECT_FORWARD_DECL(array)

class teca_binary_stream;

// trivial implementation of an array based datatset
// for testing the pipeline
class TECA_EXPORT array : public teca_dataset
{
public:

    ~array();

    /// create an array that accessible on the CPU
    static p_array new_host_accessible();

    /// create an array that accessible from CUDA
    static p_array new_cuda_accessible();

    /// create an array dataset with a specific memory memory_resource.
    ///static p_array New(const hamr::p_memory_resource &alloc);
    static p_array New(allocator alloc);

    TECA_DATASET_PROPERTY(std::vector<size_t>, extent)
    TECA_DATASET_PROPERTY(std::string, name)

    // get a pointer to the contained data that is accessible on the CPU
    std::shared_ptr<const double> get_host_accessible() const;

    // get a pointer to the contained data that is accessible from CUDA codes
    std::shared_ptr<const double> get_cuda_accessible() const;

    // check if the data is accessible from CUDA
    bool cuda_accessible() const;

    // check if the data is accessible from the CPU
    bool host_accessible() const;

    // get the raw pointer to managed data
    double *data();

    // return a unique string identifier
    std::string get_class_name() const override
    { return "array"; }

    // return an integer identifier uniquely naming the dataset type
    int get_type_code() const override
    { return 1; }

    // return true if the dataset is empty.
    bool empty() const noexcept override
    { return this->size() == 0; }

    // return a new dataset of the same type
    p_teca_dataset new_instance() const override
    { return p_teca_dataset(new array()); }

    // return a new copy constructed array
    p_teca_dataset new_copy(allocator alloc = allocator::malloc) const override;
    p_teca_dataset new_shallow_copy() override;

    size_t size() const;

    void resize(size_t n);
    void clear();

    void copy(const const_p_teca_dataset &, allocator alloc = allocator::malloc) override;
    void shallow_copy(const p_teca_dataset &) override;
    void copy_metadata(const const_p_teca_dataset &) override;
    void swap(const p_teca_dataset &) override;

    // serialize the dataset to/from the given stream
    // for I/O or communication
    int to_stream(teca_binary_stream &s) const override;
    int from_stream(teca_binary_stream &s) override;

    int to_stream(std::ostream &s) const override;
    int from_stream(std::istream &) override { return -1; }


    void debug_print() const;

#if defined(SWIG)
protected:
#else
public:
#endif
    // NOTE: constructors are public to enable std::make_shared. do not use.
    /// creates a new array dataset with the default memory resource
    array();

protected:
    /// creates a new array dataset with a specific memory resource
    ///array(const hamr::p_memory_resource &alloc);
    array(allocator alloc);

    array(const array &);
    void operator=(const array &);

private:
    std::string name;
    std::vector<size_t> extent;

    struct array_internals;
    array_internals *internals;
};

#endif
