#ifndef teca_dataset_h
#define teca_dataset_h

#include "teca_dataset_fwd.h"
#include "teca_compiler.h"
#include <iosfwd>
class teca_binary_stream;

/**
interface for teca datasets.
*/
class teca_dataset : public std::enable_shared_from_this<teca_dataset>
{
public:
    virtual ~teca_dataset() TECA_NOEXCEPT = default;

    // copy assign. this is a shallow copy
    const teca_dataset &operator=(p_teca_dataset &other)
    { this->shallow_copy(other); return *this; }

    // move assignment
    const teca_dataset &operator=(p_teca_dataset &&other)
    { this->swap(other); return *this; }

    // covert to bool. true if the dataset is not empty.
    // otherwise false.
    explicit operator bool() const TECA_NOEXCEPT
    { return !this->empty(); }

    // return true if the dataset is empty.
    virtual bool empty() const TECA_NOEXCEPT
    { return true; }

    // virtual constructor. return a new dataset of the same type.
    virtual p_teca_dataset new_instance() const = 0;

    // virtual copy constructor. return a deep copy of this
    // dataset in a new instance.
    virtual p_teca_dataset new_copy() const = 0;

    // copy data and metadata. shallow copy uses reference
    // counting, while copy duplicates the data.
    virtual void copy(const p_teca_dataset &other)
    { this->copy(other.get()); }

    virtual void copy(const teca_dataset *) = 0;

    virtual void shallow_copy(const p_teca_dataset &other)
    { this->shallow_copy(other.get()); }

    virtual void shallow_copy(const teca_dataset *) = 0;

    // copy metadata. always a deep copy.
    virtual void copy_metadata(const p_teca_dataset &other)
    { this->copy_metadata(other.get()); }

    virtual void copy_metadata(const teca_dataset *) = 0;

    // swap internals of the two objects
    virtual void swap(p_teca_dataset &other)
    { this->swap(other.get()); }

    virtual void swap(teca_dataset *) = 0;

    // serialize the dataset to/from the given stream
    // for I/O or communication
    virtual void to_stream(teca_binary_stream &) const = 0;
    virtual void from_stream(teca_binary_stream &) = 0;

    // stream to/from human readable representation
    virtual void to_stream(std::ostream &) const = 0;
    virtual void from_stream(std::istream &) {}

protected:
    teca_dataset() = default;
};

#endif
