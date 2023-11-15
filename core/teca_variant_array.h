#ifndef teca_variant_array_h
#define teca_variant_array_h

/// @file

#include <vector>
#include <string>
#include <sstream>
#include <exception>
#include <typeinfo>
#include <iterator>
#include <algorithm>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <hamr_buffer_allocator.h>

#include "teca_config.h"
#include "teca_common.h"
#include "teca_binary_stream.h"
#include "teca_bad_cast.h"
#include "teca_shared_object.h"

template <typename T> struct pod_dispatch;
template <typename T> struct object_dispatch;

TECA_SHARED_OBJECT_FORWARD_DECL(teca_variant_array)

/// A type erasure for array based data.
/** The variant array supports: set, get, assign, and append. The elements of the
 * array can be stored on different accelerator devices using different
 * technologies and accessed seamless from other accelerator devices and
 * technologies.
 *
 * Use the type erasure (this class) to implement collections of arrays and
 * public API.  Use the concrete implementation (::teca_variant_array_impl) for
 * direct access to the typed data.
 *
 * See #VARIANT_ARRAY_DISPATCH and #NESTED_VARIANT_ARRAY_DISPATCH for details
 * on how to apply type specific code to an instance of teca_variant_array.
 */
class TECA_EXPORT teca_variant_array : public std::enable_shared_from_this<teca_variant_array>
{
public:
    /// allocator types
    using allocator = hamr::buffer_allocator;

    // construct
    teca_variant_array() noexcept = default;
    virtual ~teca_variant_array() noexcept = default;

    // copy/move construct. can't copy/move construct from a base
    // pointer. these require a downcast.
    teca_variant_array(const teca_variant_array &other) = delete;
    teca_variant_array(teca_variant_array &&other) = delete;
    teca_variant_array &operator=(teca_variant_array &other) = delete;
    teca_variant_array &operator=(teca_variant_array &&other) = delete;

    /** return a newly allocated empty object of the same type. The default
     * allocator will be used.
     */
    p_teca_variant_array new_instance() const
    {
        return new_instance(allocator::malloc);
    }

    /** returns a newly allocated object of the same type will n_elem elements
     * allocated. The default allocator will be used.
     *
     *  @param[in] n_elem the number of elements to allocate.
     */
    p_teca_variant_array new_instance(size_t n) const
    {
        return new_instance(n, allocator::malloc);
    }

    /** @returns a newly allocated empty object of the same type.
     *
     * @param[in] alloc selects the allocator to use (See ::allocator)
     */
    virtual p_teca_variant_array new_instance(allocator alloc) const = 0;

    /** returns a newly allocated object of the same type will n_elem elements
     * allocated.
     *
     *  @param[in] n_elem the number of elements to allocate.
     *  @param[in[ alloc selects the allocator to use (See ::allocator)
     */
    //
    virtual p_teca_variant_array new_instance(size_t n, allocator alloc) const = 0;

    /** return a newly allocated object, initialized copy from this.
     *
     * @param[in] alloc selects the allocator to use (See ::allocator)
     */
    virtual p_teca_variant_array new_copy(allocator alloc) const = 0;

    /** return a newly allocated object, initialized copy from this. The
     * default allocator will be used.
     */
    p_teca_variant_array new_copy() const
    {
        return new_copy(allocator::malloc);
    }

    /** return a newly allocated object, initialized copy from a subset of
     * this.
     *
     * @param[in[ src_start the first element to copy
     * @param[in] n_elem the number of elements to copy
     * @param[in] alloc selects the allocator to use (See ::allocator)
     */
    virtual p_teca_variant_array new_copy(size_t src_start,
        size_t n_elem, allocator alloc) const = 0;

    /** return a newly allocated object, initialized copy from a subset of
     * this. The default allocator will be used.
     *
     * @param[in[ src_start the first element to copy
     * @param[in] n_elem the number of elements to copy
     */
    p_teca_variant_array new_copy(size_t src_start, size_t n_elem) const
    {
        return new_copy(src_start, n_elem, allocator::malloc);
    }

    /** Set the allocator. If the passed allocator is already in use this is a
     * NOOP. Otherwise the contents of the variant array are reallocated and
     * moved using a buffer allocated with the passed allocator. This can be
     * used to explicitly move the data.
     *
     * @param[in] alloc the new ::allocator to use
     * @returns 0 if successful
     */
    virtual int set_allocator(allocator alloc) = 0;

    /// return the name of the class in a human readable form
    virtual std::string get_class_name() const = 0;

    /// initialize the elements using the default constructor
    virtual void initialize() = 0;

    /** @name get
     * Copy the contest of this array to the passed instance. The desitination
     * must be large enough to hold the results.  These calls could throw
     * std::bad_cast if the passed in type is not castable to the internal
     * type.
     */
    ///@{
    /// get the contents into the other array.
    virtual void get(const p_teca_variant_array &dest) const = 0;

    /** get a subset of the contents into a subset of the other array
     *
     * @param[in] dest_start the first location to assign to
     * @param[in] src        the array to copy values from
     * @param[in] src_start  the first location to copy from
     * @param[in] n_elem     the number of elements to copy
     */
    virtual void get(size_t src_start, const p_teca_variant_array &dest,
        size_t dest_start, size_t n_elem) const = 0;

    /// get a single value
    template<typename T>
    void get(unsigned long i, T &val) const;

    /// get a single value
    template<typename T>
    T get(unsigned long i) const;

    /// get the contents into the passed vector. The vector is resized as needed.
    template<typename T>
    void get(std::vector<T> &vals) const;

    /// get a subset of the contents into the passed in array.
    template<typename T>
    void get(size_t src_start, T *dest, size_t dest_start, size_t n_elem) const;
    ///@}

    /** @name set
     * Assign values to this array form another. This array must already be
     * large enough to hold the result. If automatic resizing is desired use
     * copy instead.  These calls could throw std::bad_cast if the passed in
     * type is not castable to the internal type.
     */
    ///@{
    /// Set from the other array
    virtual void set(const const_p_teca_variant_array &src) = 0;

    /// Set from the other array
    virtual void set(const p_teca_variant_array &src)
    {
        // forward to the set from const implementation
        this->set(const_p_teca_variant_array(src));
    }

    /** Set a subset of this array from a subset opf the contents of the other
     * array.
     *
     * @param[in] dest_start the first location to assign to
     * @param[in] src        the array to copy values from
     * @param[in] src_start  the first location to copy from
     * @param[in] n_elem     the number of elements to copy
     */
    virtual void set(size_t dest_start, const const_p_teca_variant_array &src,
        size_t src_start, size_t n_elem) = 0;

    /** Set a subset of this array from a subset opf the contents of the other
     * array.
     *
     * @param[in] dest_start the first location to assign to
     * @param[in] src        the array to copy values from
     * @param[in] src_start  the first location to copy from
     * @param[in] n_elem     the number of elements to copy
     */
    virtual void set(size_t dest_start, const p_teca_variant_array &src,
        size_t src_start, size_t n_elem)
    {
        // forward to the set from const implementation
        this->set(dest_start, const_p_teca_variant_array(src), src_start, n_elem);
    }

    /// set a single value
    template<typename T>
    void set(unsigned long i, const T &val);

    /// set the contents from a vector of values. this array is not resized.
    template<typename T>
    void set(const std::vector<T> &src);

    /// set a subset of the array from a passed array. this array is not resized.
    template<typename T>
    void set(size_t dest_start, const T *src, size_t src_start, size_t n_elem);
    ///@}

    /** @name assign
     * assign the contents of the passed array to this array. This array will be
     * resized to hold the results.  These calls could throw std::bad_cast if
     * the passed in type is not castable to the internal type.
     */
    ///@{
    /// assign the contents from the other array.
    virtual void assign(const const_p_teca_variant_array &src) = 0;

    /// assign the contents from the other array.
    virtual void assign(const p_teca_variant_array &src)
    {
        // forward to the assign from const implementation
        this->assign(const_p_teca_variant_array(src));
    }

    /// assign a subset of the other array
    virtual void assign(const const_p_teca_variant_array &src,
        size_t src_start, size_t n_elem) = 0;

    /// assign a subset of the other array
    virtual void assign(const p_teca_variant_array &src,
        size_t src_start, size_t n_elem)
    {
        // forward to the assign from const implementation
        this->assign(const_p_teca_variant_array(src), src_start, n_elem);
    }

    /// assign the contents from a vector of values. this array is resized as needed.
    template<typename T>
    void assign(const std::vector<T> &src);

    /// assign the contents from a passed array. this array is resized as needed.
    template<typename T>
    void assign(const T *src, size_t src_start, size_t n_elem);

    /// copy the contents from the other array.
    virtual void copy(const const_p_teca_variant_array &src)
    {
        this->assign(src);
    }

    /// copy the contents from the other array.
    virtual void copy(const p_teca_variant_array &src)
    {
        // forward to the copy from const implementation
        this->copy(const_p_teca_variant_array(src));
    }

    /// copy a subset of the other array
    void copy(const const_p_teca_variant_array &src,
        size_t src_start, size_t n_elem)
    {
        this->assign(src, src_start, n_elem);
    }
    ///@}

    /** @name append
     * Append data at the back of the array.  These calls could throw
     * std::bad_cast if the passed in type is not castable to the internal
     * type.
     */
    ///@{
    /// append the passed array.
    virtual void append(const const_p_teca_variant_array &src) = 0;

    /// append the passed array.
    virtual void append(const p_teca_variant_array &src)
    {
        // forward to the append from const implementation
        this->append(const_p_teca_variant_array(src));
    }

    /// append a subset of the passed array
    virtual void append(const const_p_teca_variant_array &src,
        size_t src_start, size_t n_elem) = 0;

    /// append a subset of the passed array
    virtual void append(const p_teca_variant_array &src,
        size_t src_start, size_t n_elem)
    {
        // forward to the append from const implementation
        this->append(const_p_teca_variant_array(src), src_start, n_elem);
    }

    /// append a single value. this array is extended as needed.
    template<typename T>
    void append(const T &val);

    /** Append the contents from a vector of values. this array is extended as
     * needed.
     */
    template<typename T>
    void append(const std::vector<T> &src);

    /** Append the contents from a passed array. this array is extended as
     * needed.
     */
    template<typename T>
    void append(const T *src, size_t src_start, size_t n_elem);
    ///@}

    /// get the number of elements in the array
    virtual unsigned long size() const noexcept = 0;

    /// resize. allocates new storage and copies in existing values
    virtual void resize(unsigned long i) = 0;

    /** reserve. reserves the requested amount of space with out constructing
     * elements
     */
    virtual void reserve(unsigned long i) = 0;

    /// free all the internal data
    virtual void clear() noexcept = 0;

    /** swap the contents of this and the other object.  an exception is thrown
     * when no conversion between the two types exists.
     */
    virtual void swap(const p_teca_variant_array &other) = 0;

    /// compare the two arrays element wize for equality
    virtual bool equal(const const_p_teca_variant_array &other) const = 0;

    /// serrialize to the binary stream in the internal format
    virtual int to_stream(teca_binary_stream &s) const = 0;

    /// derrialize from the binary stream
    virtual int from_stream(teca_binary_stream &s) = 0;

    /// serrialize to the stream in a human readable format
    virtual int to_stream(std::ostream &s) const = 0;

    /// derrialize from the human readable stream
    virtual int from_stream(std::ostream &s) = 0;

    /// a code for the contained data type used for serialization
    virtual unsigned int type_code() const noexcept = 0;

    /// @returns true if the contents are accesisble from the CPU
    virtual int host_accessible() const noexcept = 0;

    /// @returns true if the contents are accesisble from CUDA
    virtual int cuda_accessible() const noexcept = 0;

    /// synchronize on the associated stream
    virtual void synchronize() const = 0;

private:
    template<typename T>
    void get_dispatch(unsigned long i, T &val,
        typename std::enable_if<object_dispatch<T>::value, T>::type* = 0) const;

    template<typename T>
    void get_dispatch(unsigned long i, T &val,
        typename std::enable_if<pod_dispatch<T>::value, T>::type* = 0) const;

    template<typename T>
    void get_dispatch(std::vector<T> &vals,
        typename std::enable_if<object_dispatch<T>::value, T>::type* = 0) const;

    template<typename T>
    void get_dispatch(std::vector<T> &vals,
        typename std::enable_if<pod_dispatch<T>::value, T>::type* = 0) const;

    template<typename T>
    void get_dispatch(size_t src_start, T *dest, size_t dest_start, size_t n_elem,
        typename std::enable_if<object_dispatch<T>::value, T>::type* = 0) const;

    template<typename T>
    void get_dispatch(size_t src_start, T *dest, size_t dest_start, size_t n_elem,
        typename std::enable_if<pod_dispatch<T>::value, T>::type* = 0) const;

    template<typename T>
    void set_dispatch(unsigned long i, const T &val,
        typename std::enable_if<object_dispatch<T>::value, T>::type* = 0);

    template<typename T>
    void set_dispatch(unsigned long i, const T &val,
        typename std::enable_if<pod_dispatch<T>::value, T>::type* = 0);

    template<typename T>
    void set_dispatch(const std::vector<T> &src,
        typename std::enable_if<object_dispatch<T>::value, T>::type* = 0);

    template<typename T>
    void set_dispatch(const std::vector<T> &src,
        typename std::enable_if<pod_dispatch<T>::value, T>::type* = 0);

    template<typename T>
    void set_dispatch(size_t dest_start, const T *src, size_t src_start, size_t n_elem,
        typename std::enable_if<object_dispatch<T>::value, T>::type* = 0);

    template<typename T>
    void set_dispatch(size_t dest_start, const T *src, size_t src_start, size_t n_elem,
        typename std::enable_if<pod_dispatch<T>::value, T>::type* = 0);

    template<typename T>
    void assign_dispatch(const std::vector<T> &src,
        typename std::enable_if<object_dispatch<T>::value, T>::type* = 0);

    template<typename T>
    void assign_dispatch(const std::vector<T> &src,
        typename std::enable_if<pod_dispatch<T>::value, T>::type* = 0);

    template<typename T>
    void assign_dispatch(const T *src, size_t src_start, size_t n_elem,
        typename std::enable_if<object_dispatch<T>::value, T>::type* = 0);

    template<typename T>
    void assign_dispatch(const T *src, size_t src_start, size_t n_elem,
        typename std::enable_if<pod_dispatch<T>::value, T>::type* = 0);

    template<typename T>
    void append_dispatch(const T &val,
        typename std::enable_if<object_dispatch<T>::value, T>::type* = 0);

    template<typename T>
    void append_dispatch(const T &val,
        typename std::enable_if<pod_dispatch<T>::value, T>::type* = 0);

    template<typename T>
    void append_dispatch(const std::vector<T> &src,
        typename std::enable_if<object_dispatch<T>::value, T>::type* = 0);

    template<typename T>
    void append_dispatch(const std::vector<T> &src,
        typename std::enable_if<pod_dispatch<T>::value, T>::type* = 0);

    template<typename T>
    void append_dispatch(const T *src, size_t src_start, size_t n_elem,
        typename std::enable_if<object_dispatch<T>::value, T>::type* = 0);

    template<typename T>
    void append_dispatch(const T *src, size_t src_start, size_t n_elem,
        typename std::enable_if<pod_dispatch<T>::value, T>::type* = 0);
};

#endif
