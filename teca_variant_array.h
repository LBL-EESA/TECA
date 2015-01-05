#ifndef teca_variant_array_h
#define tefca_variant_h

#include <string>
#include <exception>
#include <typeinfo>

#define teca_variant_array_gets(T) \
    virtual void get(T &v) { throw std::bad_cast(); } \
    virtual void get(unsigned int i, T &v) { throw std::bad_cast(); } \
    virtual void get(T *v) { throw std::bad_cast(); } \
    virtual void get_data(T *&d, unsigned int &s) { throw std::bad_cast(); }

#define teca_variant_array_sets(T) \
    virtual void set(const T &val) { throw std::bad_cast(); } \
    virtual void set(unsigned int i, const T &v) { throw std::bad_cast(); } \
    virtual void set(T *v) { throw std::bad_cast(); } \
    virtual void set_data(T *d, unsigned int &s) { throw std::bad_cast(); }


/// type agnostic container for simple arrays
/**
type agnostic container for array based data.
intended for use with small staticly sized
arrays, for example with meta data.
*/
class teca_variant_array
{
public:
    teca_variant_array(){}
    virtual ~teca_variant_array(){}

    teca_variant_array &operator=(teca_variant_array &other)
    { this->copy(other); }

    teca_variant_array &operator=(teca_variant_array &&other)
    { this->move(other); }

    // copy data out. an exception is thrown when
    // no conversion exists between the internal
    // type and the output type.
    teca_variant_array_gets(double)
    teca_variant_array_gets(float)
    teca_variant_array_gets(int)
    teca_variant_array_gets(unsigned int)
    teca_variant_array_gets(long)
    teca_variant_array_gets(unsigned long)
    teca_variant_array_gets(char)
    teca_variant_array_gets(std::string)

    // copy data in. an exception is thrown when
    // no conversion exists between the internal
    // type and the output type.
    teca_variant_array_sets(double)
    teca_variant_array_sets(float)
    teca_variant_array_sets(int)
    teca_variant_array_sets(unsigned int)
    teca_variant_array_sets(long)
    teca_variant_array_sets(unsigned long)
    teca_variant_array_sets(char)
    teca_variant_array_sets(std::string)

    // get the number of elements in the array
    virtual unsigned int size() = 0;

    // free all the internal data
    virtual void clear() = 0;

    // copy the contents from the other array.
    // an excpetion is thrown when no conversion
    // between the two types exists.
    virtual void copy(const teca_variant_array &other) = 0;

    // return a new'ly allocated object, initialized
    // copy from this. caller must delete.
    virtual teca_variant_array *new_copy() = 0;

    // swap the contents of this and the other object.
    // an excpetion is thrown when no conversion
    // between the two types exists.
    virtual void swap(teca_variant_array &other) = 0;

    // move the contents of the given object to this
    // object. an excpetion is thrown when no conversion
    // between the two types exists.
    virtual void move(teca_variant_array &other) = 0;

    // compare the two objects for equality
    virtual bool equal(teca_variant_array &other) = 0;
};

// implementation of our type agnistic container
// for simple arrays
template <typename T>
class teca_variant_array_impl : public teca_variant_array
{
public:
    teca_variant_array_impl() : m_data(nullptr), m_size(0) {}
    teca_variant_array_impl(unsigned int n) : m_data(new T[n]()), m_size(n) {}
    teca_variant_array_impl(T *vals, unsigned int n)
        : m_data(new T[n]()), m_size(n)
    { this->set(vals); }

    teca_variant_array_impl(const teca_variant_array_impl<T> &other)
        : m_data(nullptr) { this->copy(other); }

    teca_variant_array_impl(teca_variant_array_impl<T> &&other)
        : m_data(nullptr), m_size(0) { swap(other); }

    ~teca_variant_array_impl() { this->clear(); }

    teca_variant_array_impl<T> &
    operator=(const teca_variant_array_impl<T> &other)
    {
        this->copy(other);
        return *this;
    }

    teca_variant_array_impl<T> &
    operator=(const teca_variant_array_impl<T> &&other)
    {
        this->move(other);
        return *this;
    }

    virtual void get(T &val) { val = m_data[0]; }
    virtual void get(unsigned int i, T &val) { val = m_data[i]; }
    virtual void get(T *val)
    {
        unsigned int n = size();
        for (unsigned int i = 0; i < n; ++i)
            val[i] = m_data[i];
    }
    virtual void get_data(T *&d, unsigned int &s)
    {
        d = m_data;
        s = m_size;
    }

    virtual void set(const T &val) { m_data[0] = val; }
    virtual void set(unsigned int i, const T &val) { m_data[i] = val; }
    virtual void set(T *val)
    {
        unsigned int n = size();
        for (unsigned int i = 0; i < n; ++i)
            m_data[i] = val[i];
    }
    virtual void set_data(T *d, unsigned int s)
    {
        // NOTE:
        // call clear first if you don't want to create
        // a leak
        m_data = d;
        m_size = s;
    }

    virtual unsigned int size() { return m_size; }

    virtual bool equal(const teca_variant_array &other)
    {
        if (m_size != other.size())
            return false;

        for (unsigned int i = 0; i < m_size; ++i)
        {
            if (m_data[i] != other_data[i])
                return false;
        }
        return true;
    }

    virtual void copy(const teca_variant_array &other)
    {
        if (&other == this)
            return;

        this->clear();

        T *other_data;
        unsigned int other_size;
        other.get_data(other_data, other_size);

        m_data = new T[other_size]();
        m_size = other_size;

        for (unsigned int i = 0; i < m_size; ++i)
            m_data[i] = other_data[i];
    }

    virtual teca_variant_array *new_copy()
    {
        return new teca_variant_array_impl<T>(this);
    }

    virtual void swap(teca_variant_array &other)
    {
        T *tmp_data, *other_data;
        unsigned int tmp_size, other_size;
        this->get_data(tmp_data, tmp_size);
        other.get_data(other_data, other_size);
        other.set_data(tmp_data, tmp_size);
        this->set_data(other_data, other_size);
    }

    virtual void move(teca_variant_array &other)
    {
        this->clear();
        this->swap(other);
    }

    virtual void clear()
    {
        delete [] m_data;
        m_data = nullptr;
        m_size = 0;
    }

private:
    T *m_data;
    unsigned int m_size;
};

typedef teca_variant_array_impl<double> teca_double_array;
typedef teca_variant_array_impl<float> teca_float_array;
typedef teca_variant_array_impl<int> teca_int_array;
typedef teca_variant_array_impl<unsigned int> teca_uint_array;
typedef teca_variant_array_impl<long> teca_long_array;
typedef teca_variant_array_impl<unsigned long> teca_ulong_array;
typedef teca_variant_array_impl<char> teca_char_array;
typedef teca_variant_array_impl<std::string> teca_string_array;

#endif
