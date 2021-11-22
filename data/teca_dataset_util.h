#ifndef teca_dataset_util_h
#define teca_dataset_util_h

#include "teca_config.h"
#include "teca_dataset.h"
#include "teca_mesh.h"
#include "teca_cartesian_mesh.h"
#include "teca_uniform_cartesian_mesh.h"
#include "teca_curvilinear_mesh.h"
#include "teca_arakawa_c_grid.h"
#include "teca_table.h"
#include "teca_database.h"

/// @cond
template <typename dataset_t>
struct TECA_EXPORT teca_dataset_tt {};

template <int code>
struct TECA_EXPORT teca_dataset_new {};

#define DECLARE_DATASET_TT(_DST, _TC)   \
template <>                             \
struct teca_dataset_tt<_DST>            \
{                                       \
    enum { type_code = _TC };           \
};                                      \
                                        \
template <>                             \
struct teca_dataset_new<_TC>            \
{                                       \
    static p_ ## _DST New()             \
    { return _DST::New(); }             \
};

DECLARE_DATASET_TT(teca_table, 1)
DECLARE_DATASET_TT(teca_database, 2)
DECLARE_DATASET_TT(teca_cartesian_mesh, 3)
DECLARE_DATASET_TT(teca_uniform_cartesian_mesh, 4)
DECLARE_DATASET_TT(teca_arakawa_c_grid, 5)
DECLARE_DATASET_TT(teca_curvilinear_mesh, 6)
DECLARE_DATASET_TT(teca_array_collection, 7)

#define DATASET_FACTORY_NEW_CASE(_code)         \
    case _code:                                 \
        return teca_dataset_new<_code>::New();  \
        break;
/// @endcond


/// Constructs a new instance of teca_dataset from the provided type code.
/** The type codes are:
 *
 * | code | teca_dataset |
 * | ---- | ------------ |
 * | 1    | teca_table |
 * | 2    | teca_database |
 * | 3    | teca_cartesian_mesh |
 * | 4    | teca_uniform_cartesian_mesh |
 * | 5    | teca_arakawa_c_grid |
 * | 6    | teca_curvilinear_mesh |
 * | 7    | teca_array_collection |
 *
 */
struct TECA_EXPORT teca_dataset_factory
{
    static p_teca_dataset New(int code)
    {
        switch (code)
        {
            DATASET_FACTORY_NEW_CASE(1)
            DATASET_FACTORY_NEW_CASE(2)
            DATASET_FACTORY_NEW_CASE(3)
            DATASET_FACTORY_NEW_CASE(4)
            DATASET_FACTORY_NEW_CASE(5)
            DATASET_FACTORY_NEW_CASE(6)
            DATASET_FACTORY_NEW_CASE(7)
            default:
                TECA_ERROR("Invalid dataset code " << code )
        }
        return nullptr;
    }
};

#endif
