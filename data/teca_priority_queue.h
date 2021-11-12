#ifndef teca_priority_queue_h
#define teca_priority_queue_h

#include "teca_config.h"

#include <ostream>
#include <vector>
#include <map>
#include <type_traits>
#include <cmath>
#include <memory>


/// @cond

// use one of the following aliases for key_map_t. key_map_t
// is the type of container used to hold the locations of user provided
// keys in the heap.
// for keys that are not ordinals 0 to N, use the mapped_key_t alias
// for contiguous keys 0 to N (faster), use the contiguous_key_t alias
template<typename key_t>
using mapped_key_t = std::map<key_t, unsigned long>;

using contiguous_key_t = std::vector<unsigned long>;

// use one of the following objects to provide the priority for the
// given key. these objects internally point to the container index by
// key value holding the associated priority.
// for keys that are not ordinals 0 to N, use the mapped_key_priority_t alias
// for contiguous keys 0 to N (faster), use the contiguous_key_priority_t alias
template<typename key_t, typename priority_t>
struct TECA_EXPORT mapped_key_priority
{
    using key_map_t = mapped_key_t<key_t>;

    mapped_key_priority(std::map<key_t, priority_t> &mp) : m_map(&mp)
    {}

    priority_t operator()(key_t i)
    { return (*m_map)[i]; }

    std::map<key_t, priority_t> *m_map;
};

template<typename key_t, typename priority_t>
struct TECA_EXPORT contiguous_key_priority
{
    using key_map_t = contiguous_key_t;

    contiguous_key_priority(const std::vector<priority_t> &vec) :
        m_vec(vec.data())
    {}

    priority_t operator()(key_t i)
    { return m_vec[i]; }

    const priority_t *m_vec;
};

// forward declare the queue
template <typename key_t, typename lookup_t, typename comp_t=std::less<>,
    typename key_map_t=contiguous_key_t>
class teca_priority_queue;

// pointer type
template<typename key_t, typename lookup_t, typename ...Types >
using p_teca_priority_queue = std::shared_ptr<
    teca_priority_queue<key_t, lookup_t, Types...>>;

/// @endcond

/** @brief
 * An indirect priority queue that supports random access modification of
 * priority.
 *
 * @details
 * an indirect priority queue that supports random access modification of
 * priority the queue works with user provided keys and lookup functor that
 * converts keys to priorities.
 *
 * ### template parameters:
 *
 *  | name      | description |
 *  | ----      | ----------- |
 *  | key_t     | type of the user provided keys |
 *  | lookup_t  | callable that implements: priority_t operator()(key_t key) |
 *  | comp_t    | callable that implements the predicate: bool(key_t, key_t), |
 *  |           | used to enforce heap order. (std::less<key_t>) |
 *  | key_map_t | type of container used to track the position in the heap |
 *  |           | of the keys. The default, a vector, is only valid for |
 *  |           | interger ordinals from 0 to N. Use mapped_key_t<key_t> |
 *  |           | for all other cases. (contiguous_key_t) |
 *
 * ### typical usage:
 *
 * construct a container of objects to prioritize, and initialize a lookup
 * object that given a key returns the priority of the coresponding object.
 * create an instance of the priority_queue and push the key values. as keys
 * are pushed heap ording is imposed, this is why objects need to be in place
 * before pushing keys. when an object's priority has been changed one must
 * call modified passing the key of the object. the location of each object is
 * tracked and the queue will reprioritize itself after modification.
 *
 * ### recomendation:
 *
 * to obtain high performance, it's best to avoid using std::function for
 * lookup operations. Instead, write a small functor so that the compiler
 * can inline lookup calls.
 *
 * don't forget to change key_map_t to mapped_key_t<key_t> if
 * keys are not integer ordinals 0 to N.
 */
template <typename key_t, typename lookup_t,
    typename comp_t, typename key_map_t>
class TECA_EXPORT teca_priority_queue
{
public:

    ~teca_priority_queue() = default;

    // return a new instance, must pass the lookup operator that
    // translates keys into priority values
    static p_teca_priority_queue<key_t, lookup_t, comp_t, key_map_t>
    New(lookup_t &lookup, unsigned long init_size=256,
        unsigned long block_size=256)
    {
        p_teca_priority_queue<key_t, lookup_t,
             comp_t, key_map_t> ptr(
                new teca_priority_queue<
                    key_t, lookup_t, comp_t, key_map_t>(
                        lookup, init_size, block_size));
        return ptr;
    }

    // return true if the queue has no keys
    bool empty() { return m_end == 0; }


    /// add a value into the queue
    void push(const key_t &key);

    /// free all resources and reset the queue to an empty state
    void clear();

    // restore heap condition after an id is modified
    void modified(const key_t &key);

    // return the id at the top of the queue, and remove it.
    // internal memory is not deallocated.
    key_t pop();

    // return the id in the top of queue
    key_t peak();

    // print the state of the queue
    void to_stream(std::ostream &os, bool priorities = true);

protected:
    teca_priority_queue() = default;

    teca_priority_queue(const teca_priority_queue<key_t, lookup_t> &) = delete;
    void operator=(const teca_priority_queue<key_t, lookup_t> &) = delete;

    // initialize the queue with an comperator, the initial size, and declare
    // the amount to grow the queue by during dynamic resizing.
    template<typename u = key_map_t>
    teca_priority_queue(lookup_t lookup, unsigned long init_size,
        unsigned long block_size,
        typename std::enable_if<std::is_same<
        std::vector<unsigned long>, u>::value>::type * = 0);

    template<typename u = key_map_t>
    teca_priority_queue(lookup_t lookup, unsigned long init_size,
        unsigned long block_size,
        typename std::enable_if<std::is_same<
        std::map<key_t,unsigned long>, u>::value>::type * = 0);

    // grow the queue to the new size
    template<typename u = key_map_t>
    void grow(unsigned long n,
        typename std::enable_if<std::is_same<
        std::vector<unsigned long>, u>::value>::type * = 0);

    // grow the queue to the new size
    template<typename u = key_map_t>
    void grow(unsigned long n,
        typename std::enable_if<std::is_same<
        std::map<key_t, unsigned long>, u>::value>::type * = 0);

    // restore the heap condition starting from here
    // and working up
    void up_heapify(unsigned long id);

    // restore the heap condition starting from here
    // and working down
    void down_heapify(unsigned long id);

    // exchange two items
    void swap(unsigned long i, unsigned long j);

    // helpers for walking tree
    unsigned long left_child(unsigned long a_id)
    { return a_id*2; }

    unsigned long right_child(unsigned long a_id)
    { return a_id*2 + 1; }

    unsigned long parent(unsigned long a_id)
    { return a_id/2; }


private:
    lookup_t m_lookup;          // callable to turn keys into priority values
    std::vector<key_t> m_ids;   // array of keys
    key_map_t m_locs;           // map indexed by key to find the current position in the queue
    unsigned long m_size;       // size of the key buffer
    unsigned long m_end;        // index of the last key in the queue
    unsigned long m_block_size; // amount to grow the dynamically alloacted buffers by
};


// --------------------------------------------------------------------------
template <typename key_t, typename lookup_t,
    typename comp_t, typename key_map_t>
void teca_priority_queue<key_t, lookup_t,
    comp_t, key_map_t>::push(const key_t &key)
{
    // extend the queue
    ++m_end;

    // verify that there is space, if not allocate it
    if (m_end >= m_size)
        this->grow(m_size + m_block_size);

    // add key and it's location
    m_ids[m_end] = key;
    m_locs[key] = m_end;

    // restore heap condition
    this->up_heapify(m_end);
}

// --------------------------------------------------------------------------
template <typename key_t, typename lookup_t,
    typename comp_t, typename key_map_t>
void teca_priority_queue<key_t, lookup_t,
    comp_t, key_map_t>::clear()
{
    m_ids.clear();
    m_locs.clear();
    m_size = 0;
    m_end = 0;
}

// --------------------------------------------------------------------------
template <typename key_t, typename lookup_t,
    typename comp_t, typename key_map_t>
void teca_priority_queue<key_t, lookup_t,
    comp_t, key_map_t>::modified(const key_t &key)
{
    // find the loc of the modified key
    unsigned long id = m_locs[key];
    // fix up then down
    this->up_heapify(id);
    this->down_heapify(id);
}

// --------------------------------------------------------------------------
template <typename key_t, typename lookup_t,
    typename comp_t, typename key_map_t>
key_t teca_priority_queue<key_t, lookup_t,
    comp_t, key_map_t>::pop()
{
    key_t id_1 = m_ids[1];
    if (m_end > 0)
    {
        this->swap(1, m_end);
        --m_end;
        this->down_heapify(1);
    }
    return id_1;
}

// --------------------------------------------------------------------------
template <typename key_t, typename lookup_t,
    typename comp_t, typename key_map_t>
key_t teca_priority_queue<key_t, lookup_t,
    comp_t, key_map_t>::peak()
{
    return m_ids[1];
}

// --------------------------------------------------------------------------
template <typename key_t, typename lookup_t,
    typename comp_t, typename key_map_t>
void teca_priority_queue<key_t, lookup_t,
    comp_t, key_map_t>::to_stream(std::ostream &os, bool priorities)
{
    long log_end = std::log2(m_end);
    long n_rows = log_end + 1;
    unsigned long q = 0;
    for (long i = 0; i < n_rows; ++i)
    {
        if (q > m_end)
            break;

        long n_elem = 1 << i;
        long isp = (1 << (n_rows - 1 - i)) - 1;
        long bsp = 2*isp + 1;

        for (long j = 0; j < isp; ++j)
            os << " ";

        for (long j = 0; (j < n_elem) && (q < m_end); ++j)
        {
            if (priorities)
                os << m_lookup(m_ids[++q]);
            else
                os << m_ids[++q];
            for (long k = 0; k < bsp; ++k)
                os << " ";
        }

        os << std::endl;
    }
}

// --------------------------------------------------------------------------
template <typename key_t, typename lookup_t,
    typename comp_t, typename key_map_t>
template<typename u>
teca_priority_queue<key_t, lookup_t,
    comp_t, key_map_t>::teca_priority_queue(lookup_t lookup,
    unsigned long init_size, unsigned long block_size,
    typename std::enable_if<std::is_same<
    std::vector<unsigned long>, u>::value>::type *) :
    m_lookup(lookup), m_size(init_size), m_end(0),
    m_block_size(block_size)
{
    m_ids.resize(init_size);
    m_locs.resize(init_size);
}

// --------------------------------------------------------------------------
template <typename key_t, typename lookup_t,
    typename comp_t, typename key_map_t>
template<typename u>
teca_priority_queue<key_t, lookup_t,
    comp_t, key_map_t>::teca_priority_queue(lookup_t lookup,
    unsigned long init_size, unsigned long block_size,
    typename std::enable_if<std::is_same<
    std::map<key_t,unsigned long>, u>::value>::type *) :
    m_lookup(lookup), m_size(init_size), m_end(0),
    m_block_size(block_size)
{
    m_ids.resize(init_size);
}

// --------------------------------------------------------------------------
template <typename key_t, typename lookup_t,
    typename comp_t, typename key_map_t>
template<typename u>
void teca_priority_queue<key_t, lookup_t,
    comp_t, key_map_t>::grow(unsigned long n,
    typename std::enable_if<std::is_same<
    std::vector<unsigned long>, u>::value>::type *)
{
    m_ids.resize(n);
    m_locs.resize(n);
    m_size = n;
}

// --------------------------------------------------------------------------
template <typename key_t, typename lookup_t,
    typename comp_t, typename key_map_t>
template<typename u>
void teca_priority_queue<key_t, lookup_t,
    comp_t, key_map_t>::grow(unsigned long n,
    typename std::enable_if<std::is_same<
    std::map<key_t, unsigned long>, u>::value>::type *)
{
    m_ids.resize(n);
    m_size = n;
}


 // --------------------------------------------------------------------------
template <typename key_t, typename lookup_t,
    typename comp_t, typename key_map_t>
void teca_priority_queue<key_t, lookup_t,
    comp_t, key_map_t>::up_heapify(unsigned long id)
{
    // if at tree root then stop
    if (id < 2)
        return;

    // else find parent and enforce heap order
    comp_t comp;
    unsigned long id_p = parent(id);
    if (comp(m_lookup(m_ids[id]), m_lookup(m_ids[id_p])))
        this->swap(id, id_p);

    // continue up toward the root
    this->up_heapify(id_p);
}

// --------------------------------------------------------------------------
template <typename key_t, typename lookup_t,
    typename comp_t, typename key_map_t>
void teca_priority_queue<key_t, lookup_t,
    comp_t, key_map_t>::down_heapify(unsigned long id)
{
    // if no current node then stop
    if (id > m_end)
        return;

    // if no left child then stop
    unsigned long lc = left_child(id);
    if (lc > m_end)
        return;

    // find the smaller child
    comp_t comp;
    unsigned long smallc = lc;
    unsigned long rc = right_child(id);
    if (rc <= m_end)
        smallc = comp(m_lookup(m_ids[lc]),
            m_lookup(m_ids[rc])) ? lc : rc;

    // if in heap order then stop
    if (comp(m_lookup(m_ids[id]), m_lookup(m_ids[smallc])))
        return;

    // else swap and continue
    this->swap(id, smallc);
    this->down_heapify(smallc);
}

// --------------------------------------------------------------------------
template <typename key_t, typename lookup_t,
    typename comp_t, typename key_map_t>
void teca_priority_queue<key_t, lookup_t,
    comp_t, key_map_t>::swap(unsigned long i, unsigned long j)
{
    key_t key_i = m_ids[i];
    key_t key_j = m_ids[j];
    // exchange keys
    m_ids[i] = key_j;
    m_ids[j] = key_i;
    // update locs
    m_locs[key_j] = i;
    m_locs[key_i] = j;
}

template <typename key_t, typename lookup_t, typename ... Args>
std::ostream & operator<<(std::ostream &os, p_teca_priority_queue<key_t, lookup_t, Args ...> &q)
{
    q->to_stream(os);
    return os;
}

#endif
