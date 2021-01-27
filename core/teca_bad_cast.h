#ifndef teca_bad_cast_h
#define teca_bad_cast_h

#include <exception>
#include <string>

class teca_bad_cast : public std::exception
{
public:
    teca_bad_cast() = delete;
    ~teca_bad_cast() = default;

    teca_bad_cast(const std::string &from, const std::string &to);

    const char* what() const noexcept { return m_what.c_str(); }

private:
    std::string m_what;
};

template <typename class_t>
const std::string safe_class_name(const class_t &o)
{
    return o ? std::string(o->get_class_name()) : std::string("nullptr");
}

#endif
