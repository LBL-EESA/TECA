#ifndef teca_geography_h
#define teca_geography_h

#include <vector>
#include <string>

namespace teca_geography
{
/**
get the number of cyclone basins. cyclone basin ids are
in the range 0 to number of basins - 1.
*/
unsigned long get_number_of_cyclone_basins();

/**
get the unique list of names describing available cyclone basins.
the list can be indexed by the ids returned by the
get_cyclone_basin/s functions.
*/
void get_cyclone_basin_names(std::vector<std::string> &names,
    std::vector<std::string> &long_names);

/**
load polygons describing the cyclone basins used by TECA

upon return:

    sizes array has been appended with the size of each basin
    starts array has been appended with the starting index of
    each basin's coordinates
    x/y_coordinates have been appended with the coordinates
    ids array has been appended with the basin id

some basins are comprised of multiple polygons because
they split over the periodic boundary. hence the ids array
is used to identify a basin.
*/
void get_cyclone_basins(std::vector<unsigned long> &sizes,
    std::vector<unsigned long> &starts, std::vector<double> &x_coordinates,
    std::vector<double> &y_coordinates, std::vector<int> &ids,
    std::vector<std::string> &names, std::vector<std::string> &long_names);

/**
load a cylcone basin by name. Either the short or long name
can be used. see get_cyclone_basin_names.
*/
int get_cyclone_basin(const std::string &rname,
    std::vector<unsigned long> &sizes, std::vector<unsigned long> &starts,
    std::vector<double> &x_coordinates, std::vector<double> &y_coordinates,
    std::vector<int> &ids, std::vector<std::string> &names,
    std::vector<std::string> &long_names);

/**
load a cyclone basin by it's region id. region ids must be in the range
of 0 to get_number_of_cylone_basins() - 1.
*/
int get_cyclone_basin(unsigned int rid,
    std::vector<unsigned long> &sizes, std::vector<unsigned long> &starts,
    std::vector<double> &x_coordinates, std::vector<double> &y_coordinates,
    std::vector<int> &ids, std::vector<std::string> &names,
    std::vector<std::string> &long_names);
};

#endif
