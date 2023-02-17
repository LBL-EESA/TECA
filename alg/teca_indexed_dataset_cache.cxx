#include "teca_indexed_dataset_cache.h"

#include "teca_metadata.h"
#include "teca_metadata_util.h"
#include "teca_priority_queue.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <map>

#include <mutex>
#include <condition_variable>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

//#define TECA_DEBUG

struct cache_entry
{
    cache_entry() : m_data(nullptr), m_keep(1) {}

    std::mutex m_mutex;             // for access to the cache and time
    std::condition_variable m_cond; // use to wait for another thread to provide the data
    const_p_teca_dataset m_data;    // the dataset
    unsigned long m_keep;           // when 0 safe to delete the element
};

using p_cache_entry = std::shared_ptr<cache_entry>;

using index_t = unsigned long;
using priority_t = unsigned long;

using data_map_t = std::map<index_t, p_cache_entry>;
using use_map_t = std::map<index_t, priority_t>;

using heap_t = teca_priority_queue<index_t,   // key type (request index)
    mapped_key_priority<index_t, priority_t>, // to look up priorities
    std::less<>,                              // heapify by smallest
    mapped_key_t<index_t>>;                   // location tracking container

using p_heap_t = std::shared_ptr<heap_t>;

struct teca_indexed_dataset_cache::internals_t
{
    internals_t() : m_current_time(0)
    {
        mapped_key_priority priority_lookup(m_time_used);
        m_heap = heap_t::New(priority_lookup);
    }

    std::mutex m_mutex;         // for access to the following
    p_heap_t m_heap;            // heap with least recently used dataset at the top
    use_map_t m_time_used;      // the use time of each cached dataset
    data_map_t m_data;          // cached data
    priority_t m_current_time;  // the current time of use
};


// --------------------------------------------------------------------------
teca_indexed_dataset_cache::teca_indexed_dataset_cache() :
    max_cache_size(0), override_request_index(0), internals(new internals_t)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_indexed_dataset_cache::~teca_indexed_dataset_cache()
{
    delete this->internals;
}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_indexed_dataset_cache::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_indexed_dataset_cache":prefix));

    opts.add_options()
        TECA_POPTS_GET(unsigned long, prefix, max_cache_size,
            "Sets the maximum number of datasets to cache.")
        TECA_POPTS_GET(unsigned long, prefix, override_request_index,
            "When set always request index 0.")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_indexed_dataset_cache::set_properties(
    const std::string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, unsigned long, prefix, max_cache_size)
    TECA_POPTS_SET(opts, unsigned long, prefix, override_request_index)
}
#endif

// --------------------------------------------------------------------------
void teca_indexed_dataset_cache::clear_cache()
{
    {
    std::lock_guard<std::mutex> lock(this->internals->m_mutex);
    this->internals->m_heap->clear();
    this->internals->m_time_used.clear();
    this->internals->m_data.clear();
    this->internals->m_current_time = 0;
    }
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_indexed_dataset_cache::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_indexed_dataset_cache::get_upstream_request" << std::endl;
#endif
    (void)port;
    (void)input_md;

    std::vector<teca_metadata> up_reqs;

    // force the user to set the cache size
    if (this->max_cache_size == 0)
    {
        TECA_FATAL_ERROR("max_cache_size is 0, you must set the"
            " cache size before use.")
        return up_reqs;
    }

    // get the requested index
    index_t index = 0;
    std::string request_key;
    if (teca_metadata_util::get_requested_index(request, request_key, index))
    {
        TECA_FATAL_ERROR("Failed to get the requested index")
        return up_reqs;
    }

    // apply the override
    if (this->override_request_index)
        index = 0;

    {
    std::lock_guard<std::mutex> lock(this->internals->m_mutex);

    // is this index in the cache?
    if (this->internals->m_time_used.count(index))
    {
        // yes, update the use time
        this->internals->m_time_used[index] = ++this->internals->m_current_time;
        this->internals->m_heap->modified(index);

        // make a note that it needs to be served one more time before
        // it can be removed
        p_cache_entry elem = this->internals->m_data[index];

        {
        std::lock_guard<std::mutex> elock(elem->m_mutex);
        ++elem->m_keep;
        }

#ifdef TECA_DEBUG
        std::cerr << teca_parallel_id() << "update entry " << index
            << " keep=" << elem->m_keep << std::endl;
#endif
        if (this->get_verbose())
            TECA_STATUS("Cache hit on index " << index)

        return up_reqs;
    }

    // no, not in cache
    // set the use time and put in the heap
    this->internals->m_time_used[index] = ++this->internals->m_current_time;
    this->internals->m_heap->push(index);

    // add an empty cache enrty
    this->internals->m_data[index] = std::make_shared<cache_entry>();

#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id() << "add entry " << index << " "
        << this->internals->m_current_time << std::endl;
#endif
    }

    // generate the request for this index
    if (this->get_verbose())
        TECA_STATUS("Cache miss requesting index " << index)

    up_reqs.push_back(request);
    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_indexed_dataset_cache::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    std::cerr << teca_parallel_id()
        << "teca_indexed_dataset_cache::execute" << std::endl;
#endif
    (void)port;

    // get the requested index
    index_t index = 0;
    std::string request_key;
    if (teca_metadata_util::get_requested_index(request, request_key, index))
    {
        TECA_FATAL_ERROR("Failed to get the requested index")
        return nullptr;
    }

    // apply the override
    if (this->override_request_index)
        index = 0;

    const_p_teca_dataset data_out;

    // get the cache element associated with the requested index
    p_cache_entry elem;
    {
    std::lock_guard<std::mutex> lock(this->internals->m_mutex);;
    data_map_t::iterator it = this->internals->m_data.find(index);
    if (it == this->internals->m_data.end())
    {
        TECA_FATAL_ERROR("The cache is in an invalid state")
        return nullptr;
    }
    elem = it->second;
    }

    if (input_data.size())
    {
        // add new data to the cache
        {
        std::lock_guard<std::mutex> elock(elem->m_mutex);
        elem->m_data = input_data[0];
        --elem->m_keep;
        }
        // notify other threads that may be waiting for this data
        elem->m_cond.notify_all();
#ifdef TECA_DEBUG
        std::cerr << teca_parallel_id() << "add data " << index
            << " keep=" << elem->m_keep << std::endl;
#endif
    }
    else
    {
        // fetch existing data from the cache
        if (!elem->m_data)
        {
            // data is not yet ready, it will be provided by another thread
            std::unique_lock<std::mutex> elock(elem->m_mutex);
            if (!elem->m_data)
            {
                // data is not ready wait for another thread to provide
                elem->m_cond.wait(elock, [&]{ return bool(elem->m_data); });
                --elem->m_keep;
            }
        }
        else
        {
            // data is ready
            std::lock_guard<std::mutex> elock(elem->m_mutex);
            --elem->m_keep;
        }
#ifdef TECA_DEBUG
        std::cerr << teca_parallel_id() << "use data " << index
            << " keep=" << elem->m_keep << std::endl;
#endif
    }

    // return the dataset
    data_out = elem->m_data;

    // enforce the max cache size
    {
    std::lock_guard<std::mutex> lock(this->internals->m_mutex);
    unsigned long n_cached = this->internals->m_time_used.size();
    if (n_cached > this->max_cache_size)
    {
#ifdef TECA_DEBUG
        std::cerr << "cache too large " <<  n_cached << std::endl;
        this->internals->m_heap->to_stream(std::cerr, false);
#endif
        // might have to save some elements if they haven't been served yet
        std::vector<index_t> save;
        save.reserve(n_cached);

        unsigned long n_to_rm = n_cached - this->max_cache_size;

        // make one pass over the cache in lru order, or stop if we find
        // enough elements that can be deleted
        for (unsigned long i = 0; n_to_rm && (i < n_cached); ++i)
        {
            index_t idx = this->internals->m_heap->pop();

            p_cache_entry elem = this->internals->m_data[idx];

            // have all requests for the data been served?
            unsigned long keep = 0;
            {
            std::lock_guard<std::mutex> elock(elem->m_mutex);
            keep = elem->m_keep;
            }
            if (keep)
            {
                // no, delete later
                save.push_back(idx);
#ifdef TECA_DEBUG
                std::cerr << teca_parallel_id() << "save "
                    << idx << " keep=" << keep << std::endl;
#endif
            }
            else
            {
                // yes, delete now
                this->internals->m_data.erase(idx);
                this->internals->m_time_used.erase(idx);
                --n_to_rm;
#ifdef TECA_DEBUG
                std::cerr << teca_parallel_id() << "evict "
                    << idx << std::endl;
#endif
            }
        }

        // put elements we couldn't remove because they haven't been
        // served yet back on the heap
        unsigned long n = save.size();
        for (unsigned long i = 0; i < n; ++i)
        {
            this->internals->m_heap->push(save[i]);
        }
    }
    }

    return data_out;
}
