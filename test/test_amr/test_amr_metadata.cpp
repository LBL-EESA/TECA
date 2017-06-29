#include "teca_metadata.h"
#include "teca_variant_array.h"


#include <iostream>
#include <vector>
using namespace std;

int main(int, char **)
{
    // here is where you read global params from disk
    double dx[3] = {0.5, 0.5, 0.5};
    double x0[3] = {0.0, 0.0, 0.0};
    int ng[3] = {0, 0, 0};
    int pdom[6] = {0, 11, 0, 9, 0, 0};

    // now store the global params
    teca_metadata amr_mesh;
    amr_mesh.set("spacing", dx, 3);
    amr_mesh.set("origin", x0, 3);
    amr_mesh.set("num_ghosts", ng, 3);
    amr_mesh.set("whole_extent", pdom, 6);
    amr_mesh.set("num_levels", 2);

    vector<string> arrays;
    arrays.push_back("prw");
    arrays.push_back("temp");
    amr_mesh.set("arrays", arrays);


    // here is where you read level description from disk
    vector<vector<int>> lev0 = {
        {0,5, 0,4, 0,0}, {6,11, 0,4, 0,0},
        {0,5, 5,9, 0,0}, {6,11, 5,9, 0,0}
        };

    vector<vector<int>> lev1 = {
        {6,11, 4,9, 0,0}, {12,17, 4,9, 0,0},
        {6,11, 10,15, 0,0}, {12,17, 10,15, 0,0}
        };

    // now store the level description
    teca_metadata levels;

    teca_metadata level0;
    level0.set("ref_ratio", 1);
    level0.set("extents", lev0);
    levels.set("level_0", level0);

    teca_metadata level1;
    level1.set("ref_ratio", 2);
    level1.set("extents", lev1);
    levels.set("level_1", level1);

    amr_mesh.set("levels", levels);

    // done! let's see what we have...
    cerr << "amr_mesh = " << endl;
    amr_mesh.to_stream(cerr);
    cerr << endl;

    return 0;
}
