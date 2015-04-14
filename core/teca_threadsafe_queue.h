#ifndef teca_threadsafe_queue_h
#define teca_threadsafe_queue_h

#include <mutex>
#include <queue>
#include <condition_variable>

template<typename T>
class teca_threadsafe_queue
{
public:
    teca_threadsafe_queue() {}
    teca_threadsafe_queue(const teca_threadsafe_queue<T> &other);
    void operator=(const teca_threadsafe_queue<T> &other);

    // report current size
    typename std::queue<T>::size_type size() const;

    // push a value onto the queue
    void push(const T &val);
    void push(T &&val);

    // pop a value from the queue, will block until data is ready.
    void pop(T &val);

    // pop a value from the queue if data is present, will return
    // false if no data is in the queue.
    bool try_pop(T &val);

    // swap the contents
    void swap(teca_threadsafe_queue<T> &other);

    // clear
    void clear();

private:
    mutable std::mutex m_mutex;
    std::queue<T> m_queue;
    std::condition_variable m_ready;
    std::condition_variable m_empty;
};

// --------------------------------------------------------------------------
template<typename T>
teca_threadsafe_queue<T>::teca_threadsafe_queue(
    const teca_threadsafe_queue<T> &other)
{
    std::lock_guard<std::mutex> lock(other.m_mutex);
    std::queue<T> tmp(other.m_queue);
    m_queue.swap(tmp);
}

// --------------------------------------------------------------------------
template<typename T>
void teca_threadsafe_queue<T>::operator=(
    const teca_threadsafe_queue<T> &other)
{
    std::lock(m_mutex, other.m_mutex);
    std::lock_guard<std::mutex> lock(m_mutex, std::adopt_lock);
    std::lock_guard<std::mutex> lock_other(other.m_mutex, std::adopt_lock);
    std::queue<T> tmp(other.m_queue);
    m_queue.swap(tmp);
}

// --------------------------------------------------------------------------
template<typename T>
void teca_threadsafe_queue<T>::swap(teca_threadsafe_queue<T> &other)
{
    std::lock(m_mutex, other.m_mutex);
    std::lock_guard<std::mutex> lock(m_mutex, std::adopt_lock);
    std::lock_guard<std::mutex> lock_other(other.m_mutex, std::adopt_lock);
    m_queue.swap(other.m_queue);
}

// --------------------------------------------------------------------------
template<typename T>
typename std::queue<T>::size_type teca_threadsafe_queue<T>::size() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_queue.size();
}

// --------------------------------------------------------------------------
template<typename T>
void teca_threadsafe_queue<T>::push(const T &val)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_queue.push(val);
    m_ready.notify_one();
}

// --------------------------------------------------------------------------
template<typename T>
void teca_threadsafe_queue<T>::push(T &&val)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_queue.push(std::move(val));
    m_ready.notify_one();
}

// --------------------------------------------------------------------------
template<typename T>
void teca_threadsafe_queue<T>::pop(T &val)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    m_ready.wait(lock, [this] { return !m_queue.empty(); });
    val = std::move(m_queue.front());
    m_queue.pop();
}

// --------------------------------------------------------------------------
template<typename T>
bool teca_threadsafe_queue<T>::try_pop(T &val)
{
    std::unique_lock<std::mutex> lock(m_mutex);
    if (m_queue.empty())
        return false;
    val = std::move(m_queue.front());
    m_queue.pop();
    return true;
}

// --------------------------------------------------------------------------
template<typename T>
void teca_threadsafe_queue<T>::clear()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_queue.clear();
}

#endif
