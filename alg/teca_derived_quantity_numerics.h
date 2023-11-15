#ifndef teca_numerics_h
#define teca_numerics_h

/// @file

#include "teca_mesh.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#if !defined(SWIG)
#include "teca_variant_array_util.h"
using namespace teca_variant_array_util;
#endif

#include <string>
#include <vector>

/// Numeric code that could be reused by teca_derived_quantity
namespace teca_derived_quantity_numerics
{
/** an execute function designed for use with teca_derived_quantity
 *  on a teca_mesh. shallow copies the input and computes the
 *  point-wise average of the named variables.
 *
 *  for every i
 *  avg[i] = (v0[i] + v1[i])/2
 */
struct TECA_EXPORT point_wise_average
{
    // construct the class with two input array names, v0,v1
    // and the output array name, avg.
    point_wise_average(const std::string &v0,
        const std::string &v1, const std::string &avg)
        : m_v0(v0), m_v1(v1), m_avg(avg) {}

    point_wise_average() = delete;
    ~point_wise_average() = default;

    // implements teca_algorithm::execute
    const_p_teca_dataset operator()(unsigned int,
        const std::vector<const_p_teca_dataset> &in_data, const teca_metadata &)
    {
        const_p_teca_mesh in_mesh;
        const_p_teca_variant_array v0, v1;

        if (!(in_mesh = std::dynamic_pointer_cast<const teca_mesh>(in_data[0])) ||
            !(v0 = in_mesh->get_point_arrays()->get(m_v0)) ||
            !(v1 = in_mesh->get_point_arrays()->get(m_v1)))
        {
            TECA_ERROR("invalid inputs. mesh=" << in_mesh
                << ", " << m_v0 << "=" << v0 << ", " << m_v1 << "=" << v1)
            return nullptr;
        }

        unsigned long n_pts = v0->size();
        p_teca_variant_array avg = v0->new_instance();
        avg->resize(n_pts);

        VARIANT_ARRAY_DISPATCH(v0.get(),

            assert_type<CTT>(v1);

            auto [sp_v0, p_v0, sp_v1, p_v1] = get_host_accessible<CTT>(v0, v1);
            auto [p_avg] = data<TT>(avg);

            sync_host_access_any(v0, v1);

            for (unsigned long i = 0; i < n_pts; ++i)
                p_avg[i] = (p_v0[i] + p_v1[i])/NT(2);
            )

        p_teca_mesh out_mesh = std::static_pointer_cast<teca_mesh>(
            in_mesh->new_instance());

        out_mesh->shallow_copy(std::const_pointer_cast<teca_mesh>(in_mesh));
        out_mesh->get_point_arrays()->append(m_avg, avg);

        return out_mesh;
    }

    std::string m_v0; // input variable name 1
    std::string m_v1; // input variable name 2
    std::string m_avg; // output variable name
};

/**  an execute function designed for use with teca_derived_quantity
 *  on a teca_mesh. compute the point-wise difference of two variables
 *
 *  for every i
 *  diff[i] = v1[i] - v0[i]
 */
struct TECA_EXPORT point_wise_difference
{
    // construct the class with two input array names, v0,v1
    // and the output array name, diff.
    point_wise_difference(const std::string &v0,
        const std::string &v1, const std::string &diff)
        : m_v0(v0), m_v1(v1), m_diff(diff) {}

    point_wise_difference() = delete;
    ~point_wise_difference() = default;

    // implements teca_algorithm::execute, shallow copies the input and
    // computes the difference of two variables.
    const_p_teca_dataset operator()(unsigned int,
        const std::vector<const_p_teca_dataset> &in_data, const teca_metadata &)
    {
        const_p_teca_mesh in_mesh;
        const_p_teca_variant_array v0, v1;

        if (!(in_mesh = std::dynamic_pointer_cast<const teca_mesh>(in_data[0])) ||
            !(v0 = in_mesh->get_point_arrays()->get(m_v0)) ||
            !(v1 = in_mesh->get_point_arrays()->get(m_v1)))
        {
            TECA_ERROR("invalid inputs. mesh=" << in_mesh
                << ", " << m_v0 << "=" << v0 << ", " << m_v1 << "=" << v1)
            return nullptr;
        }

        unsigned long n_pts = v0->size();
        p_teca_variant_array diff = v0->new_instance();
        diff->resize(n_pts);

        VARIANT_ARRAY_DISPATCH(v1.get(),

            assert_type<CTT>(v1);

            auto [sp_v0, p_v0, sp_v1, p_v1] = get_host_accessible<CTT>(v0, v1);
            auto [p_diff] = data<TT>(diff);

            sync_host_access_any(v0, v1);

            for (unsigned long i = 0; i < n_pts; ++i)
                p_diff[i] = p_v1[i] - p_v0[i];
            )

        p_teca_mesh out_mesh = std::static_pointer_cast<teca_mesh>(
            in_mesh->new_instance());

        out_mesh->shallow_copy(std::const_pointer_cast<teca_mesh>(in_mesh));
        out_mesh->get_point_arrays()->append(m_diff, diff);

        return out_mesh;
    }

    std::string m_v0; // input variable name 1
    std::string m_v1; // input variable name 2
    std::string m_diff; // output variable name
};
}
#endif
