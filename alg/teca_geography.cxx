#include "teca_geography.h"


namespace teca_geography
{
// cyclone basins
// data describing the default geographic regions
// storms are sorted by
unsigned int reg_ids[] = {0, 1, 2, 3, 4, 5, 6, 7, 7, 4};
unsigned long reg_sizes[] = {5, 5, 6, 6, 14, 14, 7, 5, 7, 5};
unsigned long reg_starts[] = {0, 5, 10, 16, 22, 36, 50, 57, 62, 69};

const char *reg_names[] = {"SI", "SWP",
    "NWP", "NI", "NA", "NEP", "SEP", "SA"};

const char *reg_long_names[] = {"S Indian", "SW Pacific", "NW Pacific",
    "N Indian", "N Atlantic", "NE Pacific", "SE Pacific", "S Atlantic"};

// since we want to allow for an arbitrary
// set of polys, we can't hard code up a search
// optimization structure. but we can order them
// from most likeley+smallest to least likely+largest
double reg_lon[] = {
    // 0 S Indian (green)
    136, 136, 20, 20, 136,
    // 1 SW Pacific (pink)
    136, 216, 216, 136, 136,
    // 2 NW Pacific (orange)
    104, 180, 180, 98.75, 98.75, 104,
    // 3 N Indian (purple)
    20, 104, 98.75, 98.75, 20, 20,
    // 4 N Atlantic (red_360)
    282, 284, 284, 278, 268, 263, 237, 237, 224, 200, 200, 360.1, 360.1, 282,
    // 5 NE Pacific (yellow)
    200, 180, 180, 282, 284, 284, 278, 268, 263, 237, 237, 224, 200, 200,
    // 6 SE Pacific (cyan)
    216, 216, 289, 289, 298, 298, 216,
    // 7 S Atlantic (blue_0)
    -0.1, 20, 20, -0.1, -0.1,
    // 7 S Atlantic (blue_360)
    298, 298, 289, 289, 360.1, 360.1, 298,
    // 4 N Atlantic (red_0)
    20, 20, -0.1, -0.1, 20};

double reg_lat[] = {
    // S Indian (green)
    0, -90, -90, 0, 0,
    // SW Pacific (pink)
    -90, -90, 0, 0, -90,
    // NW Pacific (orange)
    0, 0, 90, 90, 9, 0,
    // N Indian (purple)
    0, 0, 9, 90, 90, 0,
    // N Atlantic (red_360)
    0, 3, 8.5, 8.5, 17, 17, 43, 50, 62, 62, 90, 90, 0, 0,
    // NE Pacific (yellow)
    90, 90, 0, 0, 3, 8.5, 8.5, 17, 17, 43, 50, 62, 62, 90,
    // SE Pacific (cyan)
    0, -90, -90, -52, -19.5, 0, 0,
    // S Atlantic (blue_0)
    0, 0, -90, -90, 0,
    // S Atlantic (blue_360)
    0, -19.5, -52, -90, -90, 0, 0,
    // N Atlantic (red_0)
    90, 0, 0, 90, 90};

// --------------------------------------------------------------------------
unsigned long get_number_of_cyclone_basins()
{
    return sizeof(teca_geography::reg_names)/sizeof(char*);
}

// --------------------------------------------------------------------------
int get_cyclone_basin(const std::string &rname,
    std::vector<unsigned long> &sizes, std::vector<unsigned long> &starts,
    std::vector<double> &x_coordinates, std::vector<double> &y_coordinates,
    std::vector<int> &ids, std::vector<std::string> &names,
    std::vector<std::string> &long_names)
{
    // ids array is used to look up the name, so we need to adjust
    // the internal id to account for what ever is in there already
    int next_id = names.size();

    int valid_rname = 0;
    unsigned int rid = 0;

    // scan them all since some are split across the periodic boundary
    unsigned int nregs = sizeof(teca_geography::reg_sizes)/sizeof(unsigned long);
    for (unsigned int i = 0; i < nregs; ++i)
    {
        unsigned int ii = teca_geography::reg_ids[i];
        if ((teca_geography::reg_names[ii] == rname)
            || (teca_geography::reg_long_names[ii] == rname))
        {
            rid = ii;
            valid_rname = 1;

            unsigned long rsize = teca_geography::reg_sizes[i];
            unsigned long rstart = teca_geography::reg_starts[i];

            unsigned long last_start = starts.size() ? starts.back() : 0;
            unsigned long last_size = sizes.size() ? sizes.back() : 0;

            sizes.push_back(rsize);
            starts.push_back(last_start + last_size);
            ids.push_back(next_id);

            for (unsigned long i = 0; i < rsize; ++i)
            {
                x_coordinates.push_back(teca_geography::reg_lon[rstart+i]);
                y_coordinates.push_back(teca_geography::reg_lat[rstart+i]);
            }

        }
    }

    // the caller gave a bogus name, don't change anything
    if (!valid_rname)
        return -1;

    // add the name of the region
    names.push_back(teca_geography::reg_names[rid]);
    long_names.push_back(teca_geography::reg_long_names[rid]);

    return 0;
}

// --------------------------------------------------------------------------
int get_cyclone_basin(unsigned int rid,
    std::vector<unsigned long> &sizes, std::vector<unsigned long> &starts,
    std::vector<double> &x_coordinates, std::vector<double> &y_coordinates,
    std::vector<int> &ids, std::vector<std::string> &names,
    std::vector<std::string> &long_names)
{
    // check that caller gave us a good id, if not bail now
    // change nothing.
    if (rid > teca_geography::get_number_of_cyclone_basins())
        return -1;

    // ids array is used to look up the name, so we need to adjust
    // the internal id to account for what ever is in there already
    int next_id = names.size();

    names.push_back(teca_geography::reg_names[rid]);
    long_names.push_back(teca_geography::reg_long_names[rid]);

    // scan them all since some are split across the periodic boundary
    unsigned int nregs = sizeof(teca_geography::reg_sizes)/sizeof(unsigned long);
    for (unsigned int i = 0; i < nregs; ++i)
    {
        if (teca_geography::reg_ids[i] == rid)
        {
            unsigned long rsize = teca_geography::reg_sizes[i];
            unsigned long rstart = teca_geography::reg_starts[i];

            unsigned long last_start = starts.size() ? starts.back() : 0;
            unsigned long last_size = sizes.size() ? sizes.back() : 0;

            sizes.push_back(rsize);
            starts.push_back(last_start + last_size);
            ids.push_back(next_id);

            for (unsigned long i = 0; i < rsize; ++i)
            {
                x_coordinates.push_back(teca_geography::reg_lon[rstart+i]);
                y_coordinates.push_back(teca_geography::reg_lat[rstart+i]);
            }

        }
    }

    return 0;
}

// --------------------------------------------------------------------------
void get_cyclone_basins(std::vector<unsigned long> &sizes,
    std::vector<unsigned long> &starts, std::vector<double> &x_coordinates,
    std::vector<double> &y_coordinates, std::vector<int> &ids,
    std::vector<std::string> &names, std::vector<std::string> &long_names)

{
    // ids array is used to look up the name, so we need to adjust
    // the internal id to account for what ever is in there already
    int next_id = names.size();

    unsigned int nregs = sizeof(teca_geography::reg_sizes)/sizeof(unsigned long);
    for (unsigned int i = 0; i < nregs; ++i)
    {
        sizes.push_back(teca_geography::reg_sizes[i]);
        starts.push_back(teca_geography::reg_starts[i]);
        ids.push_back(next_id+teca_geography::reg_ids[i]);
    }

    unsigned int npts = sizeof(teca_geography::reg_lon)/sizeof(double);
    for (unsigned int i = 0; i < npts; ++i)
    {
        x_coordinates.push_back(teca_geography::reg_lon[i]);
        y_coordinates.push_back(teca_geography::reg_lat[i]);
    }

    unsigned int nnames = sizeof(teca_geography::reg_names)/sizeof(char*);
    for (unsigned int i = 0; i < nnames; ++i)
    {
        names.push_back(teca_geography::reg_names[i]);
        long_names.push_back(teca_geography::reg_long_names[i]);
    }
}

// --------------------------------------------------------------------------
void get_cyclone_basin_names(std::vector<std::string> &names,
    std::vector<std::string> &long_names)
{
    unsigned int nnames = sizeof(teca_geography::reg_names)/sizeof(char*);
    for (unsigned int i = 0; i < nnames; ++i)
    {
        names.push_back(teca_geography::reg_names[i]);
        long_names.push_back(teca_geography::reg_long_names[i]);
    }
}

};
