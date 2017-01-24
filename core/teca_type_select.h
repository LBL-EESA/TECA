#ifndef teca_type_elevate_h
#define teca_type_elevate_h

namespace teca_type_select
{
// given two arguments, an elevate cast, selects
// the type of or casts to the higher precision
// type. note that given a signed and unsigned
// argument, signed type is selected.
template <typename t1, typename t2>
struct elevate {};

// given two areuments, a decay cast, selects the
// type of or casts to the lower precision
// type. note that given a signed and unsigned
// argument, unsigned type is selected.
template <typename t1, typename t2>
struct decay {};

#define teca_type_select(_class, _ret, _t1, _t2)    \
template <>                                         \
struct _class<_t1, _t2>                             \
{                                                   \
    using type = _ret;                              \
    static _ret cast(_t1 arg){ return arg; }        \
    static constexpr const char *type_name()        \
    { return #_ret; }                               \
};

#define teca_type_elevate_case(_ret, _t1, _t2)      \
teca_type_select(elevate, _ret, _t1, _t2)           \

#define teca_type_elevate_sym_case(_ret, _t1, _t2)  \
teca_type_elevate_case(_ret, _t1, _t2)              \
teca_type_elevate_case(_ret, _t2, _t1)

#define teca_type_decay_case(_ret, _t1, _t2)        \
teca_type_select(decay, _ret, _t1, _t2)             \

#define teca_type_decay_sym_case(_ret, _t1, _t2)    \
teca_type_decay_case(_ret, _t1, _t2)                \
teca_type_decay_case(_ret, _t2, _t1)

// elevate to double precision
teca_type_elevate_case(double, double, double)
teca_type_elevate_sym_case(double, double, float)
teca_type_elevate_sym_case(double, double, char)
teca_type_elevate_sym_case(double, double, short)
teca_type_elevate_sym_case(double, double, int)
teca_type_elevate_sym_case(double, double, long)
teca_type_elevate_sym_case(double, double, long long)
teca_type_elevate_sym_case(double, double, unsigned char)
teca_type_elevate_sym_case(double, double, unsigned short)
teca_type_elevate_sym_case(double, double, unsigned int)
teca_type_elevate_sym_case(double, double, unsigned long)
teca_type_elevate_sym_case(double, double, unsigned long long)
// elevate to single precision
teca_type_elevate_case(float, float, float)
teca_type_elevate_sym_case(float, float, char)
teca_type_elevate_sym_case(float, float, short)
teca_type_elevate_sym_case(float, float, int)
teca_type_elevate_sym_case(float, float, long)
teca_type_elevate_sym_case(float, float, long long)
teca_type_elevate_sym_case(float, float, unsigned char)
teca_type_elevate_sym_case(float, float, unsigned short)
teca_type_elevate_sym_case(float, float, unsigned int)
teca_type_elevate_sym_case(float, float, unsigned long)
teca_type_elevate_sym_case(float, float, unsigned long long)
// elevate to long long
teca_type_elevate_case(long long, long long, long long)
teca_type_elevate_sym_case(long long, long long, char)
teca_type_elevate_sym_case(long long, long long, short)
teca_type_elevate_sym_case(long long, long long, int)
teca_type_elevate_sym_case(long long, long long, long)
teca_type_elevate_sym_case(long long, long long, unsigned char)
teca_type_elevate_sym_case(long long, long long, unsigned short)
teca_type_elevate_sym_case(long long, long long, unsigned int)
teca_type_elevate_sym_case(long long, long long, unsigned long)
teca_type_elevate_sym_case(long long, long long, unsigned long long)
// elevate to unsigned long long
teca_type_elevate_case(unsigned long long, unsigned long long, unsigned long long)
teca_type_elevate_sym_case(long long, unsigned long long, char)  // *
teca_type_elevate_sym_case(long long, unsigned long long, short) // *
teca_type_elevate_sym_case(long long, unsigned long long, int)   // *
teca_type_elevate_sym_case(long long, unsigned long long, long)  // *
teca_type_elevate_sym_case(unsigned long long, unsigned long long, unsigned char)
teca_type_elevate_sym_case(unsigned long long, unsigned long long, unsigned short)
teca_type_elevate_sym_case(unsigned long long, unsigned long long, unsigned int)
teca_type_elevate_sym_case(unsigned long long, unsigned long long, unsigned long)
// elevate to long
teca_type_elevate_case(long, long, long)
teca_type_elevate_sym_case(long, long, char)
teca_type_elevate_sym_case(long, long, short)
teca_type_elevate_sym_case(long, long, int)
teca_type_elevate_sym_case(long, long, unsigned char)
teca_type_elevate_sym_case(long, long, unsigned short)
teca_type_elevate_sym_case(long, long, unsigned int)
teca_type_elevate_sym_case(long, long, unsigned long)
// elevate to unsigned long
teca_type_elevate_case(unsigned long, unsigned long, unsigned long)
teca_type_elevate_sym_case(long, unsigned long, char)  // *
teca_type_elevate_sym_case(long, unsigned long, short) // *
teca_type_elevate_sym_case(long, unsigned long, int)   // *
teca_type_elevate_sym_case(unsigned long, unsigned long, unsigned char)
teca_type_elevate_sym_case(unsigned long, unsigned long, unsigned short)
teca_type_elevate_sym_case(unsigned long, unsigned long, unsigned int)
// elevate to int
teca_type_elevate_case(int, int, int)
teca_type_elevate_sym_case(int, int, char)
teca_type_elevate_sym_case(int, int, short)
teca_type_elevate_sym_case(int, int, unsigned char)
teca_type_elevate_sym_case(int, int, unsigned short)
teca_type_elevate_sym_case(int, int, unsigned int)
// elevate to unsigned int
teca_type_elevate_case(unsigned int, unsigned int, unsigned int)
teca_type_elevate_sym_case(int, unsigned int, char)  // *
teca_type_elevate_sym_case(int, unsigned int, short) // *
teca_type_elevate_sym_case(unsigned int, unsigned int, unsigned char)
teca_type_elevate_sym_case(unsigned int, unsigned int, unsigned short)
// elevate to short
teca_type_elevate_case(short, short, short)
teca_type_elevate_sym_case(short, short, char)
teca_type_elevate_sym_case(short, short, unsigned char)
teca_type_elevate_sym_case(short, short, unsigned short)
// elevate to unsigned short
teca_type_elevate_case(unsigned short, unsigned short, unsigned short)
teca_type_elevate_sym_case(short, unsigned short, char)  // *
teca_type_elevate_sym_case(unsigned short, unsigned short, unsigned char)
// elevate to char
teca_type_elevate_case(char, char, char)
teca_type_elevate_sym_case(char, char, unsigned char)
// elevate to unsigned char
teca_type_elevate_case(unsigned char, unsigned char, unsigned char)

// decay from double precision
teca_type_decay_case(double, double, double)
teca_type_decay_sym_case(float, double, float)
teca_type_decay_sym_case(char, double, char)
teca_type_decay_sym_case(int, double, int)
teca_type_decay_sym_case(long, double, long)
teca_type_decay_sym_case(long long, double, long long)
teca_type_decay_sym_case(unsigned char, double, unsigned char)
teca_type_decay_sym_case(unsigned int, double, unsigned int)
teca_type_decay_sym_case(unsigned long, double, unsigned long)
teca_type_decay_sym_case(unsigned long long, double, unsigned long long)
// decay from single precision
teca_type_decay_case(float, float, float)
teca_type_decay_sym_case(char, float, char)
teca_type_decay_sym_case(int, float, int)
teca_type_decay_sym_case(long, float, long)
teca_type_decay_sym_case(long long, float, long long)
teca_type_decay_sym_case(unsigned char, float, unsigned char)
teca_type_decay_sym_case(unsigned int, float, unsigned int)
teca_type_decay_sym_case(unsigned long, float, unsigned long)
teca_type_decay_sym_case(unsigned long long, float, unsigned long long)
// decay from long long
teca_type_decay_case(long long, long long, long long)
teca_type_decay_sym_case(char, long long, char)
teca_type_decay_sym_case(int, long long, int)
teca_type_decay_sym_case(long, long long, long)
teca_type_decay_sym_case(unsigned char, long long, unsigned char)
teca_type_decay_sym_case(unsigned int, long long, unsigned int)
teca_type_decay_sym_case(unsigned long, long long, unsigned long)
teca_type_decay_sym_case(unsigned long long, long long, unsigned long long)
// decay from unsigned long long
teca_type_decay_case(unsigned long long, unsigned long long, unsigned long long)
teca_type_decay_sym_case(unsigned char, unsigned long long, char)
teca_type_decay_sym_case(unsigned int, unsigned long long, int)
teca_type_decay_sym_case(unsigned long, unsigned long long, long)
teca_type_decay_sym_case(unsigned char, unsigned long long, unsigned char)
teca_type_decay_sym_case(unsigned int, unsigned long long, unsigned int)
teca_type_decay_sym_case(unsigned long, unsigned long long, unsigned long)
// decay from long
teca_type_decay_case(long, long, long)
teca_type_decay_sym_case(char, long, char)
teca_type_decay_sym_case(int, long, int)
teca_type_decay_sym_case(unsigned char, long, unsigned char)
teca_type_decay_sym_case(unsigned int, long, unsigned int)
teca_type_decay_sym_case(unsigned long, long, unsigned long)
// decay from unsigned long
teca_type_decay_case(unsigned long, unsigned long, unsigned long)
teca_type_decay_sym_case(unsigned char, unsigned long, char)
teca_type_decay_sym_case(unsigned int, unsigned long, int)
teca_type_decay_sym_case(unsigned char, unsigned long, unsigned char)
teca_type_decay_sym_case(unsigned int, unsigned long, unsigned int)
// decay from int
teca_type_decay_case(int, int, int)
teca_type_decay_sym_case(char, int, char)
teca_type_decay_sym_case(unsigned char, int, unsigned char)
teca_type_decay_sym_case(unsigned int, int, unsigned int)
// decay from unsigned int
teca_type_decay_case(unsigned int, unsigned int, unsigned int)
teca_type_decay_sym_case(unsigned char, unsigned int, char)
teca_type_decay_sym_case(unsigned char, unsigned int, unsigned char)
// decay from char
teca_type_decay_case(char, char, char)
teca_type_decay_sym_case(unsigned char, char, unsigned char)
// decay from unsigned char
teca_type_decay_case(unsigned char, unsigned char, unsigned char)
};

#endif
