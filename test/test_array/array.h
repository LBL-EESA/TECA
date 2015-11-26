#ifndef array_h
#define array_h

#include "array_fwd.h"
#include "teca_dataset.h"

#include <string>
#include <vector>

class teca_binary_stream;

// trivial implementation of an array based datatset
// for testing the pipeline
class array : public teca_dataset
{
public:
    TECA_DATASET_STATIC_NEW(array);
    virtual ~array() = default;

    TECA_DATASET_PROPERTY(std::vector<double>, data)
    TECA_DATASET_PROPERTY(std::vector<size_t>, extent)
    TECA_DATASET_PROPERTY(std::string, name)

    // return true if the dataset is empty.
    virtual bool empty() const noexcept override
    { return this->data.empty(); }

    // return a new dataset of the same type
    virtual p_teca_dataset new_instance() const override
    { return p_teca_dataset(new array()); }

    // return a new copy constructed array
    virtual p_teca_dataset new_copy() const override;

    size_t size() const
    { return this->data.size(); }

    void resize(size_t n);
    void clear();

    double &get(size_t i)
    { return this->data[i]; }

    const double &get(size_t i) const
    { return this->data[i]; }

    double &operator[](size_t i)
    { return this->data[i]; }

    const double &operator[](size_t i) const
    { return this->data[i]; }

    void append(double d)
    { this->data.push_back(d); }

    virtual void copy(const const_p_teca_dataset &) override;
    virtual void shallow_copy(const p_teca_dataset &) override;
    virtual void copy_metadata(const const_p_teca_dataset &) override;
    virtual void swap(p_teca_dataset &) override;

    // serialize the dataset to/from the given stream
    // for I/O or communication
    virtual void to_stream(teca_binary_stream &s) const override;
    virtual void from_stream(teca_binary_stream &s) override;

    virtual void to_stream(std::ostream &s) const override;

protected:
    array() : extent({0,0}) {}

    array(const array &);
    void operator=(const array &);

private:
    std::string name;
    std::vector<size_t> extent;
    std::vector<double> data;
};

#endif
