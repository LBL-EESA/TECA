#ifndef teca_owned_future_h
#define teca_owned_future_h

#include <future>
#include <thread>

/// a future that is owned by a single thread
template <typename data_t>
struct owned_future
{
    owned_future() = delete;
    owned_future(const owned_future&) = delete;
    void operator=(const owned_future&) = delete;

    /// move construct from another instance
    owned_future(owned_future &&other) :
        m_future(std::move(other.m_future)), m_owner(other.m_owner) {}

    /// move contruct from std::future
    owned_future(std::future<data_t> &&future) :
        m_future(std::move(future)), m_owner(std::this_thread::get_id()) {}

    /// move assign form anotehr instance
    void operator=(owned_future &&other)
    {
        m_future = std::move(other.m_future);
        m_owner = other.m_owner;
    }

    /// true if this future belongs to the calling thread
    bool owner()
    {
        return std::this_thread::get_id() == m_owner;
    }

    /// access the managed future's data
    data_t &get() { return m_future.get(); }

    std::future<data_t> m_future; ///< the managed future
    std::thread::id m_owner;      ///< the thread id of the thread which created the future
};

#endif
