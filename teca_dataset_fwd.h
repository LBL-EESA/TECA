#ifndef teca_dataset_fwd_h
#define teca_dataset_fwd_h

#include "teca_common.h"
#include <memory>
#include <vector>

class teca_dataset;
using p_teca_dataset = std::shared_ptr<teca_dataset>;
using const_p_teca_dataset = std::shared_ptr<const teca_dataset>;

// this is a convenience macro to be used to
// declare New and enable seamless operation
// with std C++11 shared pointer
#define TECA_DATASET_STATIC_NEW(T)                                  \
                                                                    \
static p_##T New()                                                  \
{                                                                   \
    return p_##T(new T);                                            \
}                                                                   \
                                                                    \
using enable_shared_from_this<teca_dataset>::shared_from_this;      \
                                                                    \
std::shared_ptr<T> shared_from_this()                               \
{                                                                   \
    return std::static_pointer_cast<T>(shared_from_this());         \
}                                                                   \
                                                                    \
std::shared_ptr<T const> shared_from_this() const                   \
{                                                                   \
    return std::static_pointer_cast<T const>(shared_from_this());   \
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
#define TECA_DATASET_METADATA(key, T, len, md_obj) \
TECA_DATASET_METADATA_V(T, md_obj, key, len) \
TECA_DATASET_METADATA_A(T, md_obj, key, len) \
TECA_DATASET_METADATA_ ## len (T, md_obj, key)


#define TECA_DATASET_METADATA_1(T, md_obj, key) \
void set_##key(const T & val_1) \
{ \
    md_obj.insert<T>(#key, val_1); \
} \
int get_##key(T &val_1) \
{ \
    return md_obj.get<T>(#key, val_1); \
}

#define TECA_DATASET_METADATA_2(T, md_obj, key) \
void set_##key(const T & val_1, const T & val_2) \
{ \
    md_obj.insert<T>(#key, {val_1, val_2}); \
} \
int get_##key(T &val_1, T &val_2) \
{ \
    std::vector<T> vals; \
    if (md_obj.get<T>(#key, vals)) \
        return -1; \
    val_1 = vals[0]; \
    val_2 = vals[1]; \
    return 0; \
}

#define TECA_DATASET_METADATA_3(T, md_obj, key) \
void set_##key(const T & val_1, const T & val_2, const T & val_3) \
{ \
    md_obj.insert<T>(#key, {val_1, val_2, val_3}); \
} \
int get_##key(T &val_1, T &val_2, T &val_3) \
{ \
    std::vector<T> vals; \
    if (md_obj.get<T>(#key, vals)) \
        return -1; \
    val_1 = vals[0]; \
    val_2 = vals[1]; \
    val_3 = vals[2]; \
    return 0; \
}

#define TECA_DATASET_METADATA_4(T, md_obj, key) \
void set_##key(const T & val_1, const T & val_2, \
    const T & val_3, const T & val_4) \
{ \
    md_obj.insert<T>(#key, {val_1, val_2, val_3, val_4}); \
}

#define TECA_DATASET_METADATA_6(T, md_obj, key) \
void set_##key(const T & val_1, const T & val_2, const T & val_3, \
    const T & val_4, const T & val_5, const T & val_6) \
{ \
    md_obj.insert<T>(#key, {val_1, val_2, val_3, val_4, val_5, val_6}); \
}

#define TECA_DATASET_METADATA_8(T, md_obj, key) \
void set_##key(const T & val_1, const T & val_2, const T & val_3, const T & val_4, \
    const T & val_5, const T & val_6, const T & val_7, const T & val_8) \
{ \
    md_obj.insert<T>(#key, {val_1, val_2, val_3, val_4, val_5, val_6, val_7, val_8}); \
}

#define TECA_DATASET_METADATA_V(T, md_obj, key, len) \
void set_##key(const std::vector<T> &vals) \
{ \
    if (vals.size() != len) \
        TECA_ERROR(#key " requires " #len " values") \
    md_obj.insert<T>(#key, vals); \
} \
void get_##key(std::vector<T> &vals) \
{ \
    md_obj.get<T>(#key, vals); \
}

#define TECA_DATASET_METADATA_A(T, md_obj, key, len) \
void set_##key(const T *vals) \
{ \
    md_obj.insert<T>(#key, vals, len); \
} \
void get_##key(T *vals) \
{ \
    md_obj.get<T>(#key, vals, len); \
}

#endif
