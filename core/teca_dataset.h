#ifndef teca_dataset_h
#define teca_dataset_h

#include "teca_dataset_fwd.h"
#include <iosfwd>
class teca_binary_stream;
class teca_metadata;

/**
interface for teca datasets.
*/
class teca_dataset : public std::enable_shared_from_this<teca_dataset>
{
public:
    virtual ~teca_dataset();

    // covert to bool. true if the dataset is not empty.
    // otherwise false.
    explicit operator bool() const noexcept
    { return !this->empty(); }

    // return true if the dataset is empty.
    virtual bool empty() const noexcept
    { return true; }

    // virtual constructor. return a new dataset of the same type.
    virtual p_teca_dataset new_instance() const = 0;

    // virtual copy constructor. return a deep copy of this
    // dataset in a new instance.
    virtual p_teca_dataset new_copy() const = 0;

    // copy data and metadata. shallow copy uses reference
    // counting, while copy duplicates the data.
    virtual void copy(const const_p_teca_dataset &other);
    virtual void shallow_copy(const p_teca_dataset &other);

    // copy metadata. always a deep copy.
    virtual void copy_metadata(const const_p_teca_dataset &other);

    // swap internals of the two objects
    virtual void swap(p_teca_dataset &other);

    // access metadata
    virtual teca_metadata &get_metadata() noexcept;
    virtual const teca_metadata &get_metadata() const noexcept;
    virtual void set_metadata(const teca_metadata &md);

    // serialize the dataset to/from the given stream
    // for I/O or communication
    virtual void to_stream(teca_binary_stream &) const;
    virtual void from_stream(teca_binary_stream &);

    // stream to/from human readable representation
    virtual void to_stream(std::ostream &) const;
    virtual void from_stream(std::istream &);

protected:
    teca_dataset();

    teca_dataset(const teca_dataset &) = delete;
    teca_dataset(const teca_dataset &&) = delete;

    void operator=(const p_teca_dataset &other) = delete;
    void operator=(p_teca_dataset &&other) = delete;

    teca_metadata *metadata;
};

#endif
