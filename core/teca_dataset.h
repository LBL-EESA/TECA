#ifndef teca_dataset_h
#define teca_dataset_h

#include "teca_common.h"
#include "teca_shared_object.h"
#include "teca_variant_array.h"

#include <vector>
#include <iosfwd>

class teca_binary_stream;
class teca_metadata;

TECA_SHARED_OBJECT_FORWARD_DECL(teca_dataset)

// this is a convenience macro to be used to
// declare New and enable seamless operation
// with std C++11 shared pointer
#define TECA_DATASET_STATIC_NEW(T)                  \
                                                    \
static p_##T New()                                  \
{                                                   \
    return p_##T(new T);                            \
}                                                   \
                                                    \
std::shared_ptr<T> shared_from_this()               \
{                                                   \
    return std::static_pointer_cast<T>(             \
        teca_dataset::shared_from_this());          \
}                                                   \
                                                    \
std::shared_ptr<T const> shared_from_this() const   \
{                                                   \
    return std::static_pointer_cast<T const>(       \
        teca_dataset::shared_from_this());          \
}

// convenience macro implementing new_instance method
#define TECA_DATASET_NEW_INSTANCE()                 \
virtual p_teca_dataset new_instance() const override\
{                                                   \
    return this->New();                             \
}

// convenience macro implementing new_copy method
#define TECA_DATASET_NEW_COPY()                     \
virtual p_teca_dataset new_copy() const override    \
{                                                   \
    p_teca_dataset o = this->new_instance();        \
    o->copy(this->shared_from_this());              \
    return o;                                       \
}                                                   \
                                                    \
virtual p_teca_dataset new_shallow_copy() override  \
{                                                   \
    p_teca_dataset o = this->new_instance();        \
    o->shallow_copy(this->shared_from_this());      \
    return o;                                       \
}

// convenience macro for adding properties to dataset
// objects
#define TECA_DATASET_PROPERTY(T, name)  \
                                        \
void set_##name(const T &val)           \
{                                       \
    this->name = val;                   \
}                                       \
                                        \
const T &get_##name() const             \
{                                       \
    return this->name;                  \
}                                       \
                                        \
T &get_##name()                         \
{                                       \
    return this->name;                  \
}

// convenience set get methods for dataset metadata
#define TECA_DATASET_METADATA(key, T, len)          \
TECA_DATASET_METADATA_V(T, key, len)                \
TECA_DATASET_METADATA_A(T, key, len)                \
TECA_DATASET_METADATA_ ## len (T, key)


#define TECA_DATASET_METADATA_1(T, key)             \
void set_##key(const T & val_1)                     \
{                                                   \
    this->get_metadata().set<T>(#key, val_1);       \
}                                                   \
                                                    \
int get_##key(T &val_1) const                       \
{                                                   \
    return this->get_metadata().get<T>(             \
        #key, val_1);                               \
}

#define TECA_DATASET_METADATA_2(T, key)             \
void set_##key(const T & val_1, const T & val_2)    \
{                                                   \
    this->get_metadata().set<T>(                    \
        #key, {val_1, val_2});                      \
}                                                   \
                                                    \
int get_##key(T &val_1, T &val_2) const             \
{                                                   \
    std::vector<T> vals;                            \
    if (this->get_metadata().get<T>(#key, vals))    \
        return -1;                                  \
    val_1 = vals[0];                                \
    val_2 = vals[1];                                \
    return 0;                                       \
}

#define TECA_DATASET_METADATA_3(T, key)             \
void set_##key(const T & val_1, const T & val_2,    \
    const T & val_3)                                \
{                                                   \
    this->get_metadata().set<T>(#key,               \
        {val_1, val_2, val_3});                     \
}                                                   \
                                                    \
int get_##key(T &val_1, T &val_2, T &val_3) const   \
{                                                   \
    std::vector<T> vals;                            \
    if (this->get_metadata().get<T>(#key, vals))    \
        return -1;                                  \
    val_1 = vals[0];                                \
    val_2 = vals[1];                                \
    val_3 = vals[2];                                \
    return 0;                                       \
}

#define TECA_DATASET_METADATA_4(T, key)             \
void set_##key(const T & val_1, const T & val_2,    \
    const T & val_3, const T & val_4)               \
{                                                   \
    this->get_metadata().set<T>(#key,               \
         {val_1, val_2, val_3, val_4});             \
}

#define TECA_DATASET_METADATA_6(T, key)             \
void set_##key(const T & val_1, const T & val_2,    \
    const T & val_3, const T & val_4,               \
    const T & val_5, const T & val_6)               \
{                                                   \
    this->get_metadata().set<T>(#key,               \
        {val_1, val_2, val_3,                       \
        val_4, val_5, val_6});                      \
}

#define TECA_DATASET_METADATA_8(T, key)             \
void set_##key(const T & val_1, const T & val_2,    \
    const T & val_3, const T & val_4,               \
    const T & val_5, const T & val_6,               \
    const T & val_7, const T & val_8)               \
{                                                   \
    this->get_metadata().set<T>(#key,               \
        {val_1, val_2, val_3, val_4, val_5,         \
         val_6, val_7, val_8});                     \
}

#define TECA_DATASET_METADATA_V(T, key, len)            \
void set_##key(const std::vector<T> &vals)              \
{                                                       \
    if (vals.size() != len)                             \
    {                                                   \
        TECA_ERROR(#key " requires " #len " values")    \
    }                                                   \
    this->get_metadata().set<T>(#key, vals);            \
}                                                       \
                                                        \
int get_##key(std::vector<T> &vals) const               \
{                                                       \
    return this->get_metadata().get<T>(#key, vals);     \
}                                                       \
                                                        \
void set_##key(const p_teca_variant_array &vals)        \
{                                                       \
    if (vals->size() != len)                            \
    {                                                   \
        TECA_ERROR(#key " requires " #len " values")    \
    }                                                   \
    this->get_metadata().set(#key, vals);               \
}                                                       \
                                                        \
int get_##key(p_teca_variant_array vals) const          \
{                                                       \
    return this->get_metadata().get(#key, vals);        \
}                                                       \
                                                        \
void set_##key(const std::initializer_list<T> &l)       \
{                                                       \
    std::vector<T> vals(l);                             \
    if (vals.size() != len)                             \
    {                                                   \
        TECA_ERROR(#key " requires " #len " values")    \
    }                                                   \
    this->get_metadata().set<T>(#key, vals);            \
}                                                       \

#define TECA_DATASET_METADATA_A(T, key, len)            \
void set_##key(const T *vals)                           \
{                                                       \
    this->get_metadata().set<T>(#key, vals, len);       \
}                                                       \
                                                        \
int get_##key(T *vals) const                            \
{                                                       \
    return this->get_metadata().get<T>(                 \
        #key, vals, len);                               \
}

/// Interface for TECA datasets.
class teca_dataset : public std::enable_shared_from_this<teca_dataset>
{
public:
    virtual ~teca_dataset();

    // the name of the key that holds the index identifying this dataset
    // this should be set by the algorithm that creates the dataset.
    TECA_DATASET_METADATA(index_request_key, std::string, 1)

    // a dataset metadata that uses the value of index_request_key to
    // store the index that identifies this dataset. this should be set
    // by the algorithm that creates the dataset.
    virtual int get_request_index(long &val) const;
    virtual int set_request_index(const std::string &key, long val);
    virtual int set_request_index(long val);

    // covert to boolean. true if the dataset is not empty.
    // otherwise false.
    explicit operator bool() const noexcept
    { return !this->empty(); }

    // return true if the dataset is empty.
    virtual bool empty() const noexcept
    { return true; }

    // virtual constructor. return a new dataset of the same type.
    virtual p_teca_dataset new_instance() const = 0;

    // virtual copy constructor. return a shallow/deep copy of this
    // dataset in a new instance.
    virtual p_teca_dataset new_copy() const = 0;
    virtual p_teca_dataset new_shallow_copy() = 0;

    // return a string identifier uniquely naming the dataset type
    virtual std::string get_class_name() const = 0;

    // return an integer identifier uniquely naming the dataset type
    virtual int get_type_code() const = 0;

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
    virtual int to_stream(teca_binary_stream &) const;
    virtual int from_stream(teca_binary_stream &);

    // stream to/from human readable representation
    virtual int to_stream(std::ostream &) const;
    virtual int from_stream(std::istream &);

protected:
    teca_dataset();

    teca_dataset(const teca_dataset &) = delete;
    teca_dataset(const teca_dataset &&) = delete;

    void operator=(const p_teca_dataset &other) = delete;
    void operator=(p_teca_dataset &&other) = delete;

    teca_metadata *metadata;
};

#endif
