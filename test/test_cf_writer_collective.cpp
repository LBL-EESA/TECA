#include "teca_config.h"
#include "teca_metadata.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_array_attributes.h"
#include "teca_algorithm.h"
#include "teca_cartesian_mesh_source.h"
#include "teca_cartesian_mesh_reader.h"
#include "teca_cartesian_mesh_writer.h"
#include "teca_cf_reader.h"
#include "teca_cf_writer.h"
#include "teca_dataset_diff.h"
#include "teca_index_executive.h"
#include "teca_system_interface.h"
#include "teca_file_util.h"
#include "teca_mpi_manager.h"
#include "teca_mpi.h"
#include "teca_system_util.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <unistd.h>

class generate_test_data;
using p_generate_test_data = std::shared_ptr<generate_test_data>;

using namespace teca_variant_array_util;

// This class generates point centered data according to the function:
//
//     z = sin^2(x*y + t)
//
// Additionally the variable 'counts' holds the number of cells
// equal to or above the threshold in the first element and the number
// of cells below the threshold in the second.The variable information
// variable 'threshold' stores the threshold.
class generate_test_data : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(generate_test_data)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(generate_test_data)
    TECA_ALGORITHM_CLASS_NAME(generate_test_data)
    ~generate_test_data() {}

    TECA_ALGORITHM_PROPERTY(double, threshold)

    std::vector<std::string> get_point_array_names()
    { return {"z"}; }

    std::vector<std::string> get_info_array_names()
    { return {"counts", "z_threshold"}; }

protected:
    generate_test_data();

    using teca_algorithm::get_output_metadata;

    teca_metadata get_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    std::vector<teca_metadata> get_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    double threshold;
    int verbose;
};

// --------------------------------------------------------------------------
generate_test_data::generate_test_data() : threshold(0.5), verbose(0)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_metadata generate_test_data::get_output_metadata(unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
    (void)port;

    if (this->verbose)
        TECA_STATUS("generate_test_dataa::get_output_metadata")

    // report arrays we generate
    teca_metadata md_out = input_md[0];
    std::vector<std::string> arrays;
    md_out.get("arrays", arrays);
    arrays.push_back("z");

    // get the extent of the dataset
    long wext[6] = {0};
    if (md_out.get("whole_extent", wext, 6))
    {
        TECA_ERROR("missing whole extent")
        return teca_metadata();
    }

    long ncells = (wext[1] - wext[0] + 1)*
        (wext[3] - wext[2] + 1)*(wext[5] - wext[4] + 1);

    // create the metadata for the writer
    teca_array_attributes z_atts(
        teca_variant_array_code<double>::get(),
        teca_array_attributes::point_centering,
        ncells, teca_array_attributes::xyzt_active(), "meters", "height",
        "height is defined by the function z=sin^2(x*y + t)");

    teca_array_attributes zt_atts(
        teca_variant_array_code<double>::get(),
        teca_array_attributes::no_centering, 1, teca_array_attributes::none_active(),
        "meters", "threshold height", "value of height used to segment the z data");

    teca_array_attributes count_atts(
        teca_variant_array_code<int>::get(),
        teca_array_attributes::no_centering, 2, teca_array_attributes::none_active(),
        "cells", "number of cells", "number of cells above and below the threshold value");

    // put it in the array attributes
    teca_metadata atts;
    md_out.get("attributes", atts);
    atts.set("z", (teca_metadata)z_atts);
    atts.set("z_threshold", (teca_metadata)zt_atts);
    atts.set("counts", (teca_metadata)count_atts);
    md_out.set("attributes", atts);

    return md_out;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> generate_test_data::get_upstream_request(
    unsigned int port, const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
    (void)port;
    (void)input_md;

    if (this->verbose)
        TECA_STATUS("generate_test_dataa::get_upstream_request")

    teca_metadata up_req(request);
    std::set<std::string> arrays;
    up_req.get("arrays", arrays);
    arrays.erase("z");
    up_req.set("arrays", arrays);
    return {up_req};
}

// --------------------------------------------------------------------------
const_p_teca_dataset generate_test_data::execute(unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
    (void)port;
    (void)request;

    const_p_teca_cartesian_mesh mesh_in =
        std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[0]);

    double t = 0.0;
    mesh_in->get_time(t);

    unsigned long t_step = 0;
    mesh_in->get_time_step(t_step);

    if (this->verbose)
        TECA_STATUS("generate_test_dataa::execute time=" << t << " step=" << t_step)

    // compute sin^2(x*y + t)
    // and number of cells above and below the threshold
    unsigned long ext[6];
    mesh_in->get_extent(ext);

    unsigned long nx = ext[1] - ext[0] + 1;
    unsigned long ny = ext[3] - ext[2] + 1;
    unsigned long nxy = nx*ny;

    const_p_teca_variant_array x_in = mesh_in->get_x_coordinates();
    const_p_teca_variant_array y_in = mesh_in->get_y_coordinates();

    auto [x, px] = ::New<teca_double_array>(nx);
    auto [y, py] = ::New<teca_double_array>(ny);
    auto [z, pz] = ::New<teca_double_array>(nxy);

    double rad_per_deg = M_PI/180.0;

    unsigned long n_above = 0;

    VARIANT_ARRAY_DISPATCH(x_in.get(),

        assert_type<CTT>(y_in);

        auto [spx_in, px_in,
              spy_in, py_in] = get_host_accessible<CTT>(x_in, y_in);

        sync_host_access_any(x_in, y_in);

        // deg to rad
        for (unsigned long i = 0; i < nx; ++i)
            px[i] = px_in[i]*rad_per_deg;

        for (unsigned long i = 0; i < nx; ++i)
            py[i] = py_in[i]*rad_per_deg;

        // z = sin^2(x*y + t)
        for (unsigned long j = 0; j < ny; ++j)
        {
            unsigned long jj = nx*j;
            for (unsigned long i = 0; i < nx; ++i)
            {
                double sxyt = sin(px[i]*py[j] + t);
                pz[jj +i] = sxyt*sxyt;
            }
        }

        // count
        for (unsigned long i = 0; i < nxy; ++i)
            n_above += (pz[i] >=this->threshold ? 1 : 0);
        )

    // package up counts and thredshold
    auto [counts, p_counts] = ::New<teca_int_array>(2);
    p_counts[0] = n_above;
    p_counts[1] = nxy - n_above;

    auto [z_threshold, p_zt] = ::New<teca_double_array>(1);
    p_zt[0] = this->threshold;

    // create the output and add in the arrays
    p_teca_cartesian_mesh mesh_out = teca_cartesian_mesh::New();
    mesh_out->shallow_copy(std::const_pointer_cast<teca_cartesian_mesh>(mesh_in));
    mesh_out->get_point_arrays()->append("z", z);
    mesh_out->get_information_arrays()->append("z_threshold", z_threshold);
    mesh_out->get_information_arrays()->append("counts", counts);

    return mesh_out;
}


class print_info_arrays;
using p_print_info_arrays = std::shared_ptr<print_info_arrays>;

// This class prints the information arrays that we wrote
class print_info_arrays : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(print_info_arrays)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(print_info_arrays)
    TECA_ALGORITHM_CLASS_NAME(print_info_arrays)
    ~print_info_arrays() {}


protected:
    print_info_arrays();

    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;
};

// --------------------------------------------------------------------------
print_info_arrays::print_info_arrays()
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}


// --------------------------------------------------------------------------
const_p_teca_dataset print_info_arrays::execute(unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
    (void)port;
    (void)request;

    const_p_teca_cartesian_mesh mesh_in =
        std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[0]);

    p_teca_cartesian_mesh mesh_out = teca_cartesian_mesh::New();
    mesh_out->shallow_copy(std::const_pointer_cast<teca_cartesian_mesh>(mesh_in));

    double t = 0.0;
    mesh_out->get_time(t);

    unsigned long s = 0;
    mesh_out->get_time_step(s);

    long num_below = 0;
    long num_above = 0;

    const_p_teca_variant_array counts =
        mesh_out->get_information_arrays()->get("counts");

    counts->get(0, num_above);
    counts->get(1, num_below);

    double threshold = 0.0;

    const_p_teca_variant_array z_threshold =
        mesh_out->get_information_arrays()->get("z_threshold");

    z_threshold->get(0, threshold);

    TECA_STATUS("print_info_arrays::execute t=" << t
        << ", s=" << s << ", counts=(" << num_above << ", " << num_below
        << "), z_threshold=" << threshold)

    return mesh_out;
}




int main(int argc, char **argv)
{
    teca_mpi_manager mpi_man(argc, argv);
    int rank = mpi_man.get_comm_rank();

    teca_system_interface::set_stack_trace_on_error();
    teca_system_interface::set_stack_trace_on_mpi_error();

    if (argc != 7)
    {
        std::cerr << "test_information_array_io.py [n points] [n steps] "
            "[steps per file] [n threads] [baseline file] [baseline step]" << std::endl;
        return -1;
    }

    unsigned long nx = atoi(argv[1]);
    unsigned long n_steps = atoi(argv[2]);
    int steps_per_file = atoi(argv[3]);
    int n_threads = atoi(argv[4]);
    const char *baseline = argv[5];
    int check_step = atoi(argv[6]);

    const char *out_file = "test_cf_writer_collective-%t%.nc";
    const char *files_regex = "test_cf_writer_collective.*\\.nc$";

    // ##############
    // # Pipeline 1
    // ##############

    // construct a small mesh
    p_teca_cartesian_mesh_source src = teca_cartesian_mesh_source::New();
    src->set_whole_extents({0, nx - 1, 0, nx - 1, 0, 0, 0, n_steps-1});
    src->set_bounds({-90.0, 90.0, -90.0, 90.0, 0.0, 0.0, 0.0, 2.*M_PI});
    src->set_calendar("standard", "days since 2019-09-24");

    // generate point and information data to be written and then read
    // the point data is z = sin^2(x*y + t) thus correctness can be easily
    // verified in ParaView or ncview etc.
    p_generate_test_data gd = generate_test_data::New();
    gd->set_input_connection(src->get_output_port());

    // write the data
    p_teca_index_executive wex = teca_index_executive::New();
    wex->set_verbose(1);

    p_teca_cf_writer cfw = teca_cf_writer::New();
    cfw->set_input_connection(gd->get_output_port());
    cfw->set_verbose(1);
    cfw->set_flush_files(1);
    cfw->set_file_name(out_file);
    cfw->set_information_arrays(gd->get_info_array_names());
    cfw->set_point_arrays(gd->get_point_array_names());
    cfw->set_steps_per_file(steps_per_file);
    cfw->set_thread_pool_size(n_threads);
    cfw->set_executive(wex);
    cfw->update();

    // make sure the data makes it to disk
    MPI_Barrier(MPI_COMM_WORLD);
#if __APPLE__
    sleep(10);
#endif

    // ##############
    // # Pipeline 2
    // ##############
    if (rank == 0)
    {
        // read the data back in
        p_teca_cf_reader cfr = teca_cf_reader::New();
        cfr->set_communicator(MPI_COMM_SELF);
        cfr->set_files_regex(files_regex);
        //md = cfr->update_metadata();

        // print it out
        p_print_info_arrays par = print_info_arrays::New();
        par->set_input_connection(cfr->get_output_port());

        p_teca_index_executive rex = teca_index_executive::New();
        rex->set_arrays({"z", "z_threshold", "counts"});
        rex->set_verbose(1);
        rex->set_start_index(check_step);
        rex->set_end_index(check_step);

        std::string fn(baseline);
        teca_file_util::replace_timestep(fn, check_step);
        bool do_test = true;
        teca_system_util::get_environment_variable("TECA_DO_TEST", do_test);
        if (do_test && teca_file_util::file_exists(fn.c_str()))
        {
            std::cerr << "running the test..." << std::endl;

            p_teca_cartesian_mesh_reader cmr = teca_cartesian_mesh_reader::New();
            cmr->set_file_name(fn);

            p_teca_dataset_diff diff = teca_dataset_diff::New();
            diff->set_communicator(MPI_COMM_SELF);
            diff->set_input_connection(0, cmr->get_output_port());
            diff->set_input_connection(1, par->get_output_port());
            diff->set_executive(rex);
            diff->update();
        }
        else
        {
            std::cerr << "writing the baseline..." << std::endl;

            p_teca_cartesian_mesh_writer cmw = teca_cartesian_mesh_writer::New();
            cmw->set_communicator(MPI_COMM_SELF);
            cmw->set_file_name(baseline);
            cmw->set_input_connection(par->get_output_port());
            cmw->set_file_name(baseline);
            cmw->set_executive(rex);
            cmw->update();
        }
    }

    return 0;
}
