#ifndef teca_dataset_h
#define teca_dataset_h

#include "teca_config.h"
#include "teca_common.h"
#include "teca_shared_object.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"

#include <vector>
#include <iosfwd>
#include <type_traits>

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
    return std::make_shared<T>();                   \
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
#define TECA_DATASET_NEW_INSTANCE()                     \
virtual p_teca_dataset new_instance() const override    \
{                                                       \
    return std::make_shared                             \
        <std::remove_const<std::remove_reference        \
            <decltype(*this)>::type>::type >();         \
}

// convenience macro implementing new_copy method
#define TECA_DATASET_NEW_COPY()                         \
/** @copydoc teca_dataset::new_copy(allocator) */       \
virtual p_teca_dataset new_copy(allocator alloc =       \
    allocator::malloc) const override                   \
{                                                       \
    p_teca_dataset o = std::make_shared                 \
        <std::remove_const<std::remove_reference        \
            <decltype(*this)>::type>::type>();          \
                                                        \
    o->copy(this->shared_from_this(), alloc);           \
                                                        \
    return o;                                           \
}                                                       \
                                                        \
virtual p_teca_dataset new_shallow_copy() override      \
{                                                       \
    p_teca_dataset o = std::make_shared                 \
        <std::remove_const<std::remove_reference        \
            <decltype(*this)>::type>::type>();          \
                                                        \
    o->shallow_copy(this->shared_from_this());          \
                                                        \
    return o;                                           \
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
class TECA_EXPORT teca_dataset : public std::enable_shared_from_this<teca_dataset>
{
public:
    using allocator = teca_variant_array::allocator;

    virtual ~teca_dataset();

    /** @name index_request_key
     * The name of the key that holds the index identifying this dataset's
     * location in the index set.
     */
    ///@{
    TECA_DATASET_METADATA(index_request_key, std::string, 1)
    ///@}

    /** Set the name of the index_request_key and a key with that name set to an
     * inclusive range of indices [i0, i1].
     *
     * @param[in] request_key the name of the index_request_key will be stored
     *                        in the dataset metadata index_request_key
     * @param[in] ids an inclusive range of indices that will be stored in a
     *                key named by the value of request_key
     *
     * @returns zero if successful
     */
    virtual int set_request_indices(const std::string &request_key,
        const unsigned long ids[2]);

    /** Looks for an index_request_key and uses the value to get the inclusive
     * range of indices stored in this dataset. The call can fail if the
     * index_request_key has not been set.
     *
     * @param[out] ids the inclusive range of indices stored in this dataset.
     *
     * @returns zero if successful.
     */
    virtual int get_request_indices(unsigned long ids[2]) const;

    /** Looks for an index_request_key and uses the value to get the single index
     * stored in this dataset. The call can fail if the index_request_key has
     * not been set or if the dataset holds more than one index.
     *
     * @param[out] ids the inclusive range of indices stored in this dataset.
     *
     * @returns zero if successful.
     */
    //
    virtual int get_request_index(unsigned long &index) const;

    /** Set the name of the index_request_key and a key with that name set to an
     * inclusive range of indices [i0, i0] i.e. a single index.
     *
     * @param[in] request_key the name of the index_request_key will be stored
     *                        in the dataset metadata index_request_key
     * @param[in] ids an inclusive range of indices that will be stored in a
     *                key named by the value of request_key
     *
     * @returns zero if successful
     */
    virtual int set_request_index(const std::string &request_key,
        unsigned long index);

    /** covert to boolean. @returns true if the dataset is not empty, otherwise
     * false.
     */
    explicit operator bool() const noexcept
    { return !this->empty(); }

    /// @returns true if the dataset is empty.
    virtual bool empty() const noexcept
    { return true; }

    /// virtual constructor. return a new dataset of the same type.
    virtual p_teca_dataset new_instance() const = 0;

    /** Virtual copy constructor. return a deep copy of this dataset in a new
     * instance.
     *
     * @param[in] alloc The allocator to use for alloctions of data structures
     *                  to hold the copy. The default is a CPU based allocator.
     */
    virtual p_teca_dataset new_copy(allocator alloc = allocator::malloc) const = 0;

    /** Virtual shallow copy constructor. return a shallow copy of this dataset
     * in a new instance. References to source data structures are taken, but
     * metadata is always deep copied.
     */
    virtual p_teca_dataset new_shallow_copy() = 0;

    // return a string identifier uniquely naming the dataset type
    virtual std::string get_class_name() const = 0;

    // return an integer identifier uniquely naming the dataset type
    virtual int get_type_code() const = 0;

    /** Deep copy data and metdata.
     *
     * @param[in] other The dataset to copy.
     * @param[in] alloc The allocator to use for alloctions of data structures
     *                  to hold the copy. The default is a CPU based allocator.
     */
    virtual void copy(const const_p_teca_dataset &other,
        allocator alloc = allocator::malloc);

    /** Shallow copy data and metadata. The shallow copy takes references to
     * the source data structures. Metadata is always deep copied.
     *
     * @param[in] other The dataset to copy.
     */
    virtual void shallow_copy(const p_teca_dataset &other);

    /// copy metadata. always a deep copy.
    virtual void copy_metadata(const const_p_teca_dataset &other);

    /// swap internals of the two objects
    virtual void swap(const p_teca_dataset &other);

    /// get the dataset metadata
    virtual teca_metadata &get_metadata() noexcept;

    /// get the dataset metadata
    virtual const teca_metadata &get_metadata() const noexcept;

    /// set the dataset metadata
    virtual void set_metadata(const teca_metadata &md);

    /// serialize the dataset to the given stream for I/O or communication
    virtual int to_stream(teca_binary_stream &) const;

    /// deserialize the dataset from the given stream for I/O or communication
    virtual int from_stream(teca_binary_stream &);

    /// send to stream in a human readable representation
    virtual int to_stream(std::ostream &) const;

    /// read from stream in a human readable representation
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
