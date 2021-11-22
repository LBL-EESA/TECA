#include "teca_coordinate_util.h"
#include "teca_common.h"
#include "teca_variant_array.h"

#include <iostream>

// **************************************************************************
p_teca_double_array get_test_data(bool ascending)
{
    int nx = 10;
    double data_asc[] = {-90, -70, -50, -30, -10, 10, 30, 50, 70, 90};

    p_teca_double_array array = teca_double_array::New(nx);

    for (int i = 0; i < nx; ++i)
        array->set(i, data_asc[i] * (ascending ? 1.0 : -1.0));

    return array;
}

// **************************************************************************
void get_test_bounds(bool ascending, double *bounds)
{
    double bounds_asc[2] = {-60, 60};
    bounds[0] = bounds_asc[0] * (ascending ? 1.0 : -1.0);
    bounds[1] = bounds_asc[1] * (ascending ? 1.0 : -1.0);
}

// **************************************************************************
void get_expected_extent(bool, unsigned long *extent)
{
    unsigned long expected[2] = {1, 8};
    extent[0] = expected[0];
    extent[1] = expected[1];
}

// **************************************************************************
int check_extent(bool ascending, double *test_bounds,
    p_teca_double_array test_data, unsigned long *test_extent)
{
    // do the extent calculation
    double result_bounds[2] = {0.0, 0.0};

    test_data->get(test_extent[0], result_bounds[0]);
    test_data->get(test_extent[1], result_bounds[1]);

    std::cerr << "looked for [" << test_bounds[0] << ", "
        << test_bounds[1] << "]" << " in " << std::endl;

    test_data->to_stream(std::cerr);

    std::cerr << std::endl << "computed extent [" << test_extent[0] << ", "
        << test_extent[1] << "]" << std::endl << "result bounds [" << result_bounds[0]
        << ", " << result_bounds[1] << " ]" << std::endl << std::endl;


    // check against the expected
    unsigned long expect[2] = {0ul, 0ul};

    get_expected_extent(ascending, expect);

    if (test_extent[0] != expect[0])
    {
        TECA_ERROR("Low test_extent " << test_extent[0] << " != " << expect[0]
            << " in " << (ascending ? "ascending" : "descending")
            << " data")
        return -1;
    }

    if (test_extent[1] != expect[1])
    {
        TECA_ERROR("High test_extent " << test_extent[1] << " != " << expect[1]
            << " in " << (ascending ? "ascending" : "descending")
            << " data")
        return -1;
    }

    return 0;
}



int main(int, char **)
{
    p_teca_double_array test_data;
    double test_bounds[2];
    unsigned long test_extent[2];
    bool ascending[] = {true, false};

    for (int i = 0; i < 2; ++i)
    {
        test_data = get_test_data(ascending[i]);
        get_test_bounds(ascending[i], test_bounds);

        if (teca_coordinate_util::bounds_to_extent(test_bounds,
            test_data, test_extent))
            return -1;

        if (check_extent(ascending[i], test_bounds, test_data, test_extent))
            return -1;
    }

    // all checks pass
    return 0;
}
