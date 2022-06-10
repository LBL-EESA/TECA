/*

    This tests the IOU function with flexible tests possible via the
    commandline.
    
    The input arguments define the extent of boxes of size nx/ny to fill with
    the value 1.  The xur/yur values are the upper right coordinates to fill
    with ones, and the xlr/ylr values are the lower right coordinate.  The 1/2
    suffices refer to the first and second boxes.

    The expected_iou argument gives the value that the IOU algorithm should calculate for the given boxes (must be calculated in advance).
    
    The place_missing_at_00 argument places an arbitrary missing value at the upper left corner of the first box.  This tests the IOU routine's ability to ignore missing values
*/

#include "teca_iou.h"
#include "teca_cartesian_mesh_writer.h"
#include "teca_index_executive.h"
#include "teca_system_interface.h"
#include "teca_dataset_capture.h"
#include "teca_metadata.h"
#include "teca_dataset.h"
#include "teca_cartesian_mesh.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_dataset_source.h"
#include "teca_table.h"


#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <string>

using namespace std;

// check bounds of user-input coordinates
int assert_inbounds(const char * arg, unsigned long i, unsigned long N)
{
    if (i >= N){
        TECA_ERROR("Input argument " << arg << " with value " << i << 
        " is out of bounds; max bound is " << N)
        return -1;
    }
    return 0;
}

bool isequal(double a, double b, double epsilon)
{
    return fabs(a - b) < epsilon;
}

int main(int argc, char **argv)
{
    teca_system_interface::set_stack_trace_on_error();

    if (argc != 13)
    {
        cerr << "test_iou [nx] [ny] [xul1] [yul1] [xlr1] [ylr1] [xul2] [yul2]"
        "[xlr2] [ylr2] [expected_iou] [place_missing_at_00]" << endl;
        return -1;
    }

    unsigned long nx = atoi(argv[1]);
    unsigned long ny = atoi(argv[2]);
    unsigned long xul1 = atoi(argv[3]);
    unsigned long yul1 = atoi(argv[4]);
    unsigned long xlr1 = atoi(argv[5]);
    unsigned long ylr1 = atoi(argv[6]);
    unsigned long xul2 = atoi(argv[7]);
    unsigned long yul2 = atoi(argv[8]);
    unsigned long xlr2 = atoi(argv[9]);
    unsigned long ylr2 = atoi(argv[10]);
    double expected_val = atof(argv[11]);
    unsigned int do_place_missing = atoi(argv[12]);

    // check that input arguments are within bounds
    if (assert_inbounds("xul1", xul1, nx)) return -1;
    if (assert_inbounds("yul1", yul1, ny)) return -1;
    if (assert_inbounds("xlr1", xlr1, nx)) return -1;
    if (assert_inbounds("ylr1", ylr1, ny)) return -1;
    if (assert_inbounds("xul2", xul2, nx)) return -1;
    if (assert_inbounds("yul2", yul2, ny)) return -1;
    if (assert_inbounds("xlr2", xlr2, nx)) return -1;
    if (assert_inbounds("ylr2", ylr2, ny)) return -1;

    // allocate a mesh
    // coordinate axes
    using coord_t = double;
    coord_t dx = coord_t(360.)/coord_t(nx - 1);
    p_teca_variant_array_impl<coord_t> x = 
        teca_variant_array_impl<coord_t>::New(nx);
    auto spx = x->get_cpu_accessible();
    coord_t *px = spx.get();
    for (unsigned long i = 0; i < nx; ++i)
        px[i] = i*dx;

    coord_t dy = coord_t(180.)/coord_t(ny - 1);
    p_teca_variant_array_impl<coord_t> y = 
        teca_variant_array_impl<coord_t>::New(ny);
    auto spy = y->get_cpu_accessible();
    coord_t *py = spy.get();
    for (unsigned long i = 0; i < ny; ++i)
        py[i] = coord_t(-90.) + i*dy;

    p_teca_variant_array_impl<coord_t> z = 
        teca_variant_array_impl<coord_t>::New(1);
    z->set(0, 0.f);

    p_teca_variant_array_impl<coord_t> t = 
        teca_variant_array_impl<coord_t>::New(1);
    t->set(0, 1.f);

    unsigned long nxy = nx * ny;
    p_teca_double_array grid_0 = teca_double_array::New(nxy);
    auto sp_grid_0 = grid_0->get_cpu_accessible();
    double *p_grid_0 = sp_grid_0.get();
    p_teca_double_array grid_1 = teca_double_array::New(nxy);
    auto sp_grid_1 = grid_1->get_cpu_accessible();
    double *p_grid_1 = sp_grid_1.get();

    // initialize the grids
    for (unsigned int n = 0; n < nxy; ++n)
    {
        p_grid_0[n] = 0;
        p_grid_1[n] = 0;
    }


    // fill in the block in grid 0
    for (unsigned int i = xul1; i <= xlr1; ++i)
    {
        for (unsigned int j = yul1; j <= ylr1; ++j)
        {
            unsigned int n = j*nx + i;
            p_grid_0[n] = 1;
        }
    }

    // fill in the block in grid 1
    for (unsigned int i = xul2; i <= xlr2; ++i)
    {
        for (unsigned int j = yul2; j <= ylr2; ++j)
        {
            unsigned int n = j*nx + i;
            p_grid_1[n] = 1;
        }
    }

    // set the upper corner of grid 0 to the missing value if flagged
    double dFillValue = -1e20;
    if (do_place_missing) p_grid_0[0] = dFillValue;

#define DEBUG 1
// print out the tables
#ifdef DEBUG
    // print the block in grid 0
    std::cout << "grid_0:" << std::endl << std::endl;
    for (unsigned int j = 0; j < ny; ++j)
    {
        for (unsigned int i = 0; i < nx; ++i)
        {
            unsigned int n = j*nx + i;
            std::cout << " " << p_grid_0[n];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // print the block in grid 1
    std::cout << "grid_1:" << std::endl << std::endl;
    for (unsigned int j = 0; j < ny; ++j)
    {
        for (unsigned int i = 0; i < nx; ++i)
        {
            unsigned int n = j*nx + i;
            std::cout << " " << p_grid_1[n];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
#endif

    unsigned long wext[] = {0, nx - 1, 0, ny - 1, 0, 0};

    p_teca_cartesian_mesh mesh = teca_cartesian_mesh::New();
    mesh->set_x_coordinates("x", x);
    mesh->set_y_coordinates("y", y);
    mesh->set_z_coordinates("z", z);
    mesh->set_whole_extent(wext);
    mesh->set_extent(wext);
    mesh->set_time(1.0);
    mesh->set_time_step(0ul);
    mesh->get_point_arrays()->append("grid_0", grid_0);
    mesh->get_point_arrays()->append("grid_1", grid_1);

    teca_metadata md;
    md.set("whole_extent", wext, 6);
    md.set("time_steps", std::vector<unsigned long>({0}));
    md.set("variables", std::vector<std::string>({"grid_0","grid_1"}));
    md.set("number_of_time_steps", 1);
    md.set("index_initializer_key", std::string("number_of_time_steps"));
    md.set("index_request_key", std::string("time_step"));

    // build the pipeline
    p_teca_dataset_source source = teca_dataset_source::New();
    source->set_metadata(md);
    source->set_dataset(mesh);

    p_teca_iou iou = teca_iou::New();
    iou->set_input_connection(source->get_output_port());
    iou->set_iou_field_0_variable("grid_0");
    iou->set_iou_field_1_variable("grid_1");
    iou->set_fill_val_0(dFillValue);

    p_teca_index_executive exe = teca_index_executive::New();
    exe->set_start_index(0);
    exe->set_end_index(0);

    p_teca_dataset_capture iou_o = teca_dataset_capture::New();
    iou_o->set_input_connection(iou->get_output_port());
    iou_o->set_executive(exe);

    iou_o->update();

    const_p_teca_dataset ds = iou_o->get_dataset();
    // get the input
    const_p_teca_table iou_table = std::dynamic_pointer_cast<const teca_table>(ds);
    const_p_teca_variant_array iou_array = iou_table->get_column("iou");

    using TT = teca_variant_array_impl<double>;
    using NT = double;

    auto sp_iou = dynamic_cast<const TT*>(iou_array.get())->get_cpu_accessible();
    const NT *p_iou = sp_iou.get();

    // check whether the calculated IOU value matches the expectation
    NT test_val = p_iou[0];
    if (!isequal(test_val, expected_val, 1e-7))
    {
        TECA_ERROR("Value " << test_val << " is not " << expected_val)
        return -1;
    }

    return 0;
}
