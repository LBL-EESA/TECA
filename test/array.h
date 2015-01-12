#ifndef array_h
#define array_h

#include "teca_dataset.h"

#include <vector>
#include <memory>

class array;
typedef std::shared_ptr<array> p_array;

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
