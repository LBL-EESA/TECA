#ifndef teca_saffir_simpson
#define teca_saffir_simpson

#include <limits>

namespace teca_saffir_simpson
{
// Saffir-Simpson scale prescribes the following limits:
// CAT wind km/h
// -1:   0- 63  :  Tropical depression
//  0:  63-119  :  Tropical storm
//  1: 119-153 km/h
//  2: 154-177 km/h
//  3: 178-208 km/h
//  4: 209-251 km/h
//  5: 252 km/h or higher
constexpr double low_wind_bound_kmph[] = {0.0, 63.0,
    119.0, 154.0, 178.0, 209.0, 252.0};

// get the high bound for the given class of storm
constexpr double high_wind_bound_kmph[] = {63.0,
    119.0, 154.0, 178.0, 209.0, 252.0,
    std::numeric_limits<double>::max()};

template<typename n_t>
constexpr n_t get_lower_bound_kmph(int c)
{
    return low_wind_bound_kmph[++c];
}

template<typename n_t>
constexpr n_t get_upper_bound_kmph(int c)
{
    return high_wind_bound_kmph[++c];
}

// given wind speed in km/hr return Saffir-Simpson category
// NOTE: there is some ambiguity in the above as
// it's defined using integers. we are not converting
// to integer here.
// get the low bound for the given class of storm
template<typename n_t>
int classify_kmph(n_t w)
{
    if (w < n_t(high_wind_bound_kmph[0]))
        return -1;
    else
    if (w < n_t(high_wind_bound_kmph[1]))
        return 0;
    else
    if (w < n_t(high_wind_bound_kmph[2]))
        return 1;
    else
    if (w < n_t(high_wind_bound_kmph[3]))
        return 2;
    else
    if (w < n_t(high_wind_bound_kmph[4]))
        return 3;
    else
    if (w < n_t(high_wind_bound_kmph[5]))
        return 4;
    return 5;
}

// get the low bound for the given class of storm
template<typename n_t>
constexpr n_t get_lower_bound_mps(int c)
{
    return get_lower_bound_kmph<n_t>(c)/n_t(3.6);
}

// get the high bound for the given class of storm
template<typename n_t>
constexpr n_t get_upper_bound_mps(int c)
{
    return get_upper_bound_kmph<n_t>(c)/n_t(3.6);
}

// given wind speed in km/hr return Saffir-Simpson category
// NOTE: there is some ambiguity in the above as
// it's defined using integers. we are not converting
// to integer here.
// get the low bound for the given class of storm
template<typename n_t>
int classify_mps(n_t w)
{
    // 1 m/s -> 3.6 Km/h
    if (w < n_t(high_wind_bound_kmph[0]/3.6))
        return -1;
    else
    if (w < n_t(high_wind_bound_kmph[1]/3.6))
        return 0;
    else
    if (w < n_t(high_wind_bound_kmph[2]/3.6))
        return 1;
    else
    if (w < n_t(high_wind_bound_kmph[3]/3.6))
        return 2;
    else
    if (w < n_t(high_wind_bound_kmph[4]/3.6))
        return 3;
    else
    if (w < n_t(high_wind_bound_kmph[5]/3.6))
        return 4;
    return 5;
}
};

#endif
