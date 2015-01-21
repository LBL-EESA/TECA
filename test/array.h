#ifndef array_h
#define array_h

#include "teca_dataset.h"

#include <vector>
#include <memory>

class array;
typedef std::shared_ptr<array> p_array;
class teca_binary_stream;

// trivial implementation of an array based datatset
// for testing the pipeline
class array : public teca_dataset
{
public:
    TECA_DATASET_STATIC_NEW(array);
    ~array() {}

    TECA_DATASET_PROPERTY(std::vector<double>, data)
    TECA_DATASET_PROPERTY(std::vector<size_t>, extent)
    TECA_DATASET_PROPERTY(std::string, name)

    // return true if the dataset is empty.
    virtual bool empty() noexcept override
    { return this->data.empty(); }

    // return a new dataset of the same type
    virtual p_teca_dataset new_instance() override
    { return p_teca_dataset(new array); }

    // serialize the dataset to/from the given stream
    // for I/O or communication
    virtual void to_stream(teca_binary_stream &s) override;
    virtual void from_stream(teca_binary_stream &s) override;

    void copy_structure(const p_array &other)
    {
        this->name = other->name;
        this->extent = other->extent;
        this->data.resize(extent[1]-extent[0]);
    }

    void copy_data(const p_array &other)
    { data = other->data; }

    size_t size() const
    { return this->data.size(); }

    void resize(size_t n)
    {
        this->data.resize(n, 0.0);
        this->extent = {0, n};
    }

    void clear()
    {
        this->data.clear();
        this->extent[0] = 0;
        this->extent[1] = 0;
    }

    double &operator[](size_t i)
    { return this->data[i]; }

    void append(double d)
    { this->data.push_back(d); }

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
