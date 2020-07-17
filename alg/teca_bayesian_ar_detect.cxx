#include "teca_bayesian_ar_detect.h"

#include "teca_mesh.h"
#include "teca_array_collection.h"
#include "teca_array_attributes.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"
#include "teca_cartesian_mesh.h"
#include "teca_table.h"
#include "teca_binary_stream.h"
#include "teca_dataset_source.h"
#include "teca_latitude_damper.h"
#include "teca_binary_segmentation.h"
#include "teca_connected_components.h"
#include "teca_2d_component_area.h"
#include "teca_component_area_filter.h"
#include "teca_programmable_algorithm.h"
#include "teca_programmable_reduce.h"
#include "teca_dataset_capture.h"
#include "teca_index_executive.h"
#include "teca_thread_pool.h"
#include "teca_mpi.h"

#include <algorithm>
#include <iostream>
#include <deque>
#include <set>
#define _USE_MATH_DEFINES
#include <cmath>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

namespace {

// This routine appends the contents of dataset_0.get_metadata.get(property_name)
// onto that from dataset_0 and overwrites the contents `property_name' in the
// metadata of mesh_out.
//
//   property_name : (std::string) the key of the metadata property in
//                   dataset_* on which to do the reduction
//
//   dataset_0     : (p_teca_dataset) the LHS dataset in the reduction
//
//   dataset_1     : (p_teca_dataset) the RHS dataset in the reduction
//
//   mesh_out      : (p_teca_cartesian_mesh) the output of the reduction
/*void property_reduce(std::string property_name,
    p_teca_dataset dataset_0, p_teca_dataset dataset_1,
    p_teca_cartesian_mesh mesh_out)
{
    // declare the LHS and RHS property vectors
    std::vector<double> property_vector_0;
    std::vector<double> property_vector_1;

    // get the property vectors from the metadata of the LHS and RHS datasets
    dataset_0->get_metadata().get(property_name, property_vector_0);
    dataset_1->get_metadata().get(property_name, property_vector_1);

    // construct the output property vector by concatenating LHS and RHS vectors
    std::vector<double> property_vector(property_vector_0);
    property_vector.insert(property_vector.end(), property_vector_1.begin(),
        property_vector_1.end());

    // Overwrite the concatenated property vector in the output dataset
    mesh_out->get_metadata().set(property_name, property_vector);
}*/

// drive the pipeline execution once for each parameter table row
// injects the parameter values into the upstream requests
class parameter_table_request_generator
{
public:
    parameter_table_request_generator() = delete;

    parameter_table_request_generator(unsigned long n,
        const const_p_teca_variant_array &hwhm_lat_col,
        const const_p_teca_variant_array &min_water_vapor_col,
        const const_p_teca_variant_array &min_area_col) :
        parameter_table_size(n), hwhm_latitude_column(hwhm_lat_col),
        min_water_vapor_column(min_water_vapor_col),
        min_component_area_column(min_area_col)
    {}

    ~parameter_table_request_generator() = default;
    parameter_table_request_generator(const parameter_table_request_generator &) = default;

    unsigned long parameter_table_size;
    const_p_teca_variant_array hwhm_latitude_column;
    const_p_teca_variant_array min_water_vapor_column;
    const_p_teca_variant_array min_component_area_column;

    // sets up the map-reduce over the parameter table.
    // the algorithm then intercepts these keys in upstream request
    // and loads the coresponiding row of the parameter table, into
    // the request. upstream algorithms find and use the parameters.
    void initialize_index_executive(teca_metadata &md)
    {
        md.set("index_initializer_key", std::string("number_of_rows"));
        md.set("index_request_key", std::string("row_id"));
        md.set("number_of_rows", this->parameter_table_size);
    }

    // get upstream request callback that pulls a row from the parameter
    // table and puts it in the request
    std::vector<teca_metadata> operator()(unsigned int,
        const std::vector<teca_metadata> &, const teca_metadata &req)
    {
        std::vector<teca_metadata> up_reqs;

        // figure out which row of the parameter table is being requested
        if (!req.has("row_id"))
        {
            TECA_ERROR("Missing index key row_id")
            return up_reqs;
        }

        long row_id = 0;
        req.get("row_id", row_id);

        teca_metadata up_req(req);

        // get that row of the table and put in the right keys in the request
        //
        // half_width_at_half_max -- consumed by the teca_latitude_damper
        // low_threshold_value -- consumed by the teca_binary_segmentation
        // low_area_threshold -- consumed by the teca_component_area_filter
        //
        double hwhm = 0.0;
        this->hwhm_latitude_column->get(row_id, hwhm);
        up_req.set("center", 0.0);
        up_req.set("half_width_at_half_max", hwhm);

        double percentile = 0.0;
        this->min_water_vapor_column->get(row_id, percentile);
        up_req.set("low_threshold_value", percentile);

        double min_area = 0.0;
        this->min_component_area_column->get(row_id, min_area);
        up_req.set("low_area_threshold", min_area);

        up_reqs.push_back(up_req);

        return up_reqs;
    }
};

// does the reduction of each pipeline execution over each parameter
// table row
class parameter_table_reduction
{
public:
    parameter_table_reduction() = delete;

    parameter_table_reduction(unsigned long n_params,
        const std::string &comp_array_name, const std::string &prob_array_name) :
        parameter_table_size(n_params), component_array_name(comp_array_name),
        probability_array_name(prob_array_name)
    {}

    parameter_table_reduction(const parameter_table_reduction &) = default;

    ~parameter_table_reduction() = default;

    // finalize callback
    // completes the reduction by scaling by the number of parameter table rows
    p_teca_dataset operator()(const const_p_teca_dataset &ds)
    {
        p_teca_cartesian_mesh out_mesh =
            std::dynamic_pointer_cast<teca_cartesian_mesh>(ds->new_instance());

        out_mesh->shallow_copy(std::const_pointer_cast<teca_dataset>(ds));

        p_teca_variant_array ar_prob_in =
            out_mesh->get_point_arrays()->get(this->probability_array_name);

        if (!ar_prob_in)
        {
            TECA_ERROR("finalize failed, proability array \""
                << this->probability_array_name << "\" not found")
            return nullptr;
        }

        p_teca_variant_array ar_prob = ar_prob_in->new_copy();

        unsigned long n_vals = ar_prob->size();

        TEMPLATE_DISPATCH_FP(teca_variant_array_impl,
            ar_prob.get(),

            NT num_params = this->parameter_table_size;

            NT *p_ar_prob = std::static_pointer_cast<TT>(ar_prob)->get();

            for (unsigned long i = 0; i < n_vals; ++i)
                p_ar_prob[i] /= num_params;
            )

        out_mesh->get_point_arrays()->set(
            this->probability_array_name, ar_prob);

        return out_mesh;
    }

    // this reduction computes the probability from each parameter table run
    // if the inputs have the probability array this is used, if not the
    // array is computed from the filtered connected components. after the
    // reduction runs, the result will need to be normalized.
    p_teca_dataset operator()(const const_p_teca_dataset &left,
        const const_p_teca_dataset &right)
    {
        // the inputs will not be modified. we are going to make shallow
        // copy, and add an array
        p_teca_dataset dataset_0 = std::const_pointer_cast<teca_dataset>(left);
        p_teca_dataset dataset_1 = std::const_pointer_cast<teca_dataset>(right);

        p_teca_variant_array prob_out;
        p_teca_variant_array n_wvcc_out;
        p_teca_variant_array pt_row_out;

        if (dataset_0 && dataset_1)
        {
            // both inputs have data to process
            p_teca_cartesian_mesh mesh_0 = std::dynamic_pointer_cast<teca_cartesian_mesh>(dataset_0);
            p_teca_variant_array wvcc_0 = mesh_0->get_point_arrays()->get(this->component_array_name);
            p_teca_variant_array prob_0 = mesh_0->get_point_arrays()->get(this->probability_array_name);
            p_teca_variant_array n_wvcc_0 = mesh_0->get_information_arrays()->get("ar_count");
            p_teca_variant_array pt_row_0 = mesh_0->get_information_arrays()->get("parameter_table_row");
            const teca_metadata &md_0 = mesh_0->get_metadata();

            p_teca_cartesian_mesh mesh_1 = std::dynamic_pointer_cast<teca_cartesian_mesh>(dataset_1);
            p_teca_variant_array wvcc_1 = mesh_1->get_point_arrays()->get(this->component_array_name);
            p_teca_variant_array prob_1 = mesh_1->get_point_arrays()->get(this->probability_array_name);
            p_teca_variant_array n_wvcc_1 = mesh_1->get_information_arrays()->get("ar_count");
            p_teca_variant_array pt_row_1 = mesh_1->get_information_arrays()->get("parameter_table_row");
            const teca_metadata &md_1 = mesh_1->get_metadata();

            if (prob_0 && prob_1)
            {
                // both inputs already have probablilty computed, reduction takes
                // their sum
                unsigned long n_vals = prob_0->size();
                prob_out = prob_0->new_copy();
                TEMPLATE_DISPATCH_FP(teca_variant_array_impl,
                    prob_out.get(),
                    const NT *p_prob_1 = std::static_pointer_cast<TT>(prob_1)->get();
                    NT *p_prob_out = std::static_pointer_cast<TT>(prob_out)->get();
                    for (unsigned long i = 0; i < n_vals; ++i)
                        p_prob_out[i] += p_prob_1[i];
                    )

                // concatenate ar couunt and parameter table row
                n_wvcc_out = n_wvcc_0->new_copy();
                n_wvcc_out->append(*n_wvcc_1.get());

                pt_row_out = pt_row_0->new_copy();
                pt_row_out->append(*pt_row_1.get());
            }
            else if (prob_0 || prob_1)
            {
                // one of the inputs has probability computed. add the computed
                // values from the other.
                p_teca_variant_array wvcc;
                p_teca_variant_array prob;

                if (prob_0)
                {
                    // dataset 0 has valid values, dataset 1 has new raw data
                    // keep prob_0, append calc from wvcc_1
                    prob = prob_0;
                    wvcc = wvcc_1;

                    // append ar count
                    // don't count the background label as an ar detection.
                    // TECA by convention uses label 0 for cells out side of
                    // the segmentation (i.e. the background), and by
                    // convention this is stored in the first entry.
                    int val = 0;
                    if (md_1.get("component_ids", &val, 1))
                    {
                        TECA_ERROR("mesh 1 is missing component_ids")
                        return nullptr;
                    }
                    int wvcc_bg = val == 0 ? 1 : 0;

                    if (md_1.get("number_of_components", val))
                    {
                        TECA_ERROR("mesh 1 is missing number_of_components")
                        return nullptr;
                    }

                    n_wvcc_out = n_wvcc_0->new_copy();
                    n_wvcc_out->append(val - wvcc_bg);

                    // append param table row
                    if (md_1.get("parameter_table_row", val))
                    {
                        TECA_ERROR("mesh 1 is missing parameter_table_row")
                        return nullptr;
                    }

                    pt_row_out = pt_row_0->new_copy();
                    pt_row_out->append(val);
                }
                else
                {
                    // dataset 1 has valid values, dataset 0 has new raw data
                    // keep prob_1, append calc from wvcc_0
                    prob = prob_1;
                    wvcc = wvcc_0;

                    // append ar count
                    // don't count the background label as an ar detection.
                    // TECA by convention uses label 0 for cells out side of
                    // the segmentation (i.e. the background), and by
                    // convention this is stored in the first entry.
                    int val = 0;
                    if (md_0.get("component_ids", &val, 1))
                    {
                        TECA_ERROR("mesh 1 is missing component_ids")
                        return nullptr;
                    }
                    int wvcc_bg = val == 0 ? 1 : 0;

                    if (md_0.get("number_of_components", val))
                    {
                        TECA_ERROR("mesh 0 is missing number_of_components")
                        return nullptr;
                    }

                    n_wvcc_out = n_wvcc_1->new_copy();
                    n_wvcc_out->append(val - wvcc_bg);

                    // append param table row
                    if (md_0.get("parameter_table_row", val))
                    {
                        TECA_ERROR("mesh 0 is missing parameter_table_row")
                        return nullptr;
                    }

                    pt_row_out = pt_row_1->new_copy();
                    pt_row_out->append(val);
                }

                if (!wvcc)
                {
                    TECA_ERROR("pipeline error, component array \""
                        << this->component_array_name << "\" is not present")
                    return nullptr;
                }

                // do the calculation
                unsigned long n_vals = prob->size();
                prob_out = prob->new_copy();

                NESTED_TEMPLATE_DISPATCH_I(teca_variant_array_impl,
                    wvcc.get(),
                    _COMP,

                    const NT_COMP *p_wvcc = std::static_pointer_cast<TT_COMP>(wvcc)->get();

                    NESTED_TEMPLATE_DISPATCH_FP(teca_variant_array_impl,
                        prob_out.get(),
                        _PROB,

                        NT_PROB *p_prob_out = std::static_pointer_cast<TT_PROB>(prob_out)->get();

                        for (unsigned long i = 0; i < n_vals; ++i)
                            p_prob_out[i] += (p_wvcc[i] > 0 ? NT_PROB(1) : NT_PROB(0));
                        )
                    )
            }
            else
            {
                // neither input has probability computed, compute from the filtered
                // connected components.
                if (!wvcc_0 || !wvcc_1)
                {
                    TECA_ERROR("pipeline error, component array \""
                        << this->component_array_name << "\" is not present")
                    return nullptr;
                }

                unsigned long n_vals = wvcc_0->size();

                prob_out = teca_float_array::New();
                prob_out->resize(n_vals);

                NESTED_TEMPLATE_DISPATCH_I(teca_variant_array_impl,
                    wvcc_0.get(),
                    _COMP,

                    const NT_COMP *p_wvcc_0 = std::static_pointer_cast<TT_COMP>(wvcc_0)->get();
                    const NT_COMP *p_wvcc_1 = std::static_pointer_cast<TT_COMP>(wvcc_1)->get();

                    NESTED_TEMPLATE_DISPATCH_FP(teca_variant_array_impl,
                        prob_out.get(),
                        _PROB,
                        NT_PROB *p_prob_out = std::static_pointer_cast<TT_PROB>(prob_out)->get();
                        for (unsigned long i = 0; i < n_vals; ++i)
                            p_prob_out[i] = (p_wvcc_0[i] > 0 ? NT_PROB(1) : NT_PROB(0)) +
                                 (p_wvcc_1[i] > 0 ? NT_PROB(1) : NT_PROB(0));
                        )
                    )

                // append ar count
                int vals[2];
                if (md_0.get("number_of_components", vals[0]) ||
                    md_1.get("number_of_components", vals[1]))
                {
                    TECA_ERROR("number_of_components missing from mesh 0 and/or mesh 1")
                    return nullptr;
                }

                // don't count the background label as an ar detection.
                // TECA by convention uses label 0 for cells out side of
                // the segmentation (i.e. the background), and by
                // convention this is stored in the first entry.
                int val = 0;
                if (md_0.get("component_ids", &val, 1))
                {
                    TECA_ERROR("mesh 1 is missing component_ids")
                    return nullptr;
                }
                int wvcc_bg = val == 0 ? 1 : 0;
                vals[0] -= wvcc_bg;

                if (md_1.get("component_ids", &val, 1))
                {
                    TECA_ERROR("mesh 1 is missing component_ids")
                    return nullptr;
                }
                wvcc_bg = val == 0 ? 1 : 0;
                vals[1] -= wvcc_bg;

                n_wvcc_out = teca_int_array::New(vals, 2);

                // append param table row
                if (md_0.get("parameter_table_row", vals[0]) ||
                    md_1.get("parameter_table_row", vals[1]))
                {
                    TECA_ERROR("paramteter_table_row missing from mesh 0 and/or mesh 1")
                    return nullptr;
                }

                pt_row_out = teca_int_array::New(vals, 2);
            }
        }
        else if (dataset_0 || dataset_1)
        {
            // only one of the inputs has data to process.
            p_teca_cartesian_mesh mesh = dataset_0 ?
                std::dynamic_pointer_cast<teca_cartesian_mesh>(dataset_0) :
                std::dynamic_pointer_cast<teca_cartesian_mesh>(dataset_1);

            p_teca_variant_array prob =
                mesh->get_point_arrays()->get(this->probability_array_name);

            p_teca_variant_array n_wvcc =
                mesh->get_information_arrays()->get("ar_count");

            p_teca_variant_array pt_row =
                mesh->get_information_arrays()->get("parameter_table_row");

            const teca_metadata &md = mesh->get_metadata();

            if (prob)
            {
                // probability has already been computed, pass it through
                prob_out = prob;
                n_wvcc_out = n_wvcc;
                pt_row_out = pt_row;
            }
            else
            {
                // compute the probability from the connected components
                p_teca_variant_array wvcc =
                    mesh->get_point_arrays()->get(this->component_array_name);
                if (wvcc)
                {
                    TECA_ERROR("pipeline error, component array \""
                        << this->component_array_name << "\" is not present")
                    return nullptr;
                }

                unsigned long n_vals = wvcc->size();

                prob_out = teca_float_array::New();
                prob_out->resize(n_vals);

                NESTED_TEMPLATE_DISPATCH_I(teca_variant_array_impl,
                    wvcc.get(),
                    _COMP,

                    const NT_COMP *p_wvcc = std::static_pointer_cast<TT_COMP>(wvcc)->get();

                    NESTED_TEMPLATE_DISPATCH_FP(teca_variant_array_impl,
                        prob_out.get(),
                        _PROB,

                        NT_PROB *p_prob_out = std::static_pointer_cast<TT_PROB>(prob_out)->get();

                        for (unsigned long i = 0; i < n_vals; ++i)
                            p_prob_out[i] = (p_wvcc[i] > 0 ? NT_PROB(1) : NT_PROB(0));
                        )
                    )

                // get ar counts from metadata and pass into the information
                // arrays
                // don't count the background label as an ar detection.
                // TECA by convention uses label 0 for cells out side of
                // the segmentation (i.e. the background), and by
                // convention this is stored in the first entry.
                int val = 0;
                if (md.get("component_ids", &val, 1))
                {
                    TECA_ERROR("mesh 1 is missing component_ids")
                    return nullptr;
                }
                int wvcc_bg = val == 0 ? 1 : 0;

                if (md.get("number_of_components", val))
                {
                    TECA_ERROR("metadata missing number_of_components")
                    return nullptr;
                }

                val -= wvcc_bg;

                n_wvcc_out = teca_int_array::New(&val, 1);

                // get parameter table rows from metadata and pass into the
                // information arrays
                if (md.get("parameter_table_row", val))
                {
                    TECA_ERROR("meatdata missing parameter_table_row")
                    return nullptr;
                }

                pt_row_out = teca_int_array::New(&val,1);
            }
        }
        else
        {
            // neither input has valid dataset, this should not happen
            TECA_ERROR("nothing to reduce, must have at least 1 dataset")
            return nullptr;
        }

        // construct the output, set the probability array. this will be the
        // only array, but all metadata is passed through.
        p_teca_cartesian_mesh mesh_out = teca_cartesian_mesh::New();

        if (dataset_0)
            mesh_out->copy_metadata(dataset_0);
        else if (dataset_1)
            mesh_out->copy_metadata(dataset_1);

        mesh_out->get_point_arrays()->append(this->probability_array_name, prob_out);

        mesh_out->get_information_arrays()->append("ar_count", n_wvcc_out);
        mesh_out->get_information_arrays()->append("parameter_table_row", pt_row_out);

        return mesh_out;
    }

private:
    unsigned long parameter_table_size;
    std::string component_array_name;   // input
    std::string probability_array_name; // output
};

}



// PIMPL idiom hides internals
struct teca_bayesian_ar_detect::internals_t
{
    internals_t();
    ~internals_t();

    void clear();

    teca_algorithm_output_port parameter_pipeline_port; // pipeline that serves up tracks
    const_p_teca_table parameter_table;                 // parameter table
    teca_metadata metadata;                             // cached metadata
    p_teca_data_request_queue queue;                    // thread pool
};

// --------------------------------------------------------------------------
teca_bayesian_ar_detect::internals_t::internals_t()
{}

// --------------------------------------------------------------------------
teca_bayesian_ar_detect::internals_t::~internals_t()
{}

// --------------------------------------------------------------------------
void teca_bayesian_ar_detect::internals_t::clear()
{
    this->metadata.clear();
    this->parameter_table = nullptr;
}

// --------------------------------------------------------------------------
teca_bayesian_ar_detect::teca_bayesian_ar_detect() :
    min_component_area_variable("min_component_area"),
    min_water_vapor_variable("min_water_vapor"),
    hwhm_latitude_variable("hwhm_latitude"), thread_pool_size(1),
    verbose(0), internals(new internals_t)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_bayesian_ar_detect::~teca_bayesian_ar_detect()
{
    delete this->internals;
}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_bayesian_ar_detect::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_bayesian_ar_detect":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, water_vapor_variable,
            "name of the water vapor variable (\"\")")
        TECA_POPTS_GET(std::string, prefix, min_component_area_variable,
            "name of the column in the parameter table containing the "
            "component area threshold (\"min_component_area\")")
        TECA_POPTS_GET(std::string, prefix, min_water_vapor_variable,
            "name of the column in the parameter table containing the "
            "water vapor threshold (\"min_water_vapor\")")
        TECA_POPTS_GET(std::string, prefix, hwhm_latitude_variable,
            "name of the column in the parameter table containing the "
            "half width at half max latitude (\"hwhm_latitude\")")
        TECA_POPTS_GET(int, prefix, thread_pool_size,
            "number of threads to parallelize execution over (1)")
        TECA_POPTS_GET(int, prefix, verbose,
            "flag indicating diagnostic info should be displayed in "
            "the terminal (0)")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_bayesian_ar_detect::set_properties(const std::string &prefix,
    variables_map &opts)
{
    TECA_POPTS_SET(opts, std::string, prefix, water_vapor_variable)
    TECA_POPTS_SET(opts, std::string, prefix, min_component_area_variable)
    TECA_POPTS_SET(opts, std::string, prefix, min_water_vapor_variable)
    TECA_POPTS_SET(opts, std::string, prefix, hwhm_latitude_variable)
    TECA_POPTS_SET(opts, int, prefix, thread_pool_size)
    TECA_POPTS_SET(opts, int, prefix, verbose)
}
#endif

// --------------------------------------------------------------------------
void teca_bayesian_ar_detect::set_input_connection(unsigned int id,
        const teca_algorithm_output_port &port)
{
    if (id == 0)
        this->internals->parameter_pipeline_port = port;
    else
        this->teca_algorithm::set_input_connection(0, port);
}

// --------------------------------------------------------------------------
void teca_bayesian_ar_detect::set_modified()
{
    // clear cached metadata before forwarding on to
    // the base class.
    this->internals->clear();
    teca_algorithm::set_modified();
}

// --------------------------------------------------------------------------
void teca_bayesian_ar_detect::set_thread_pool_size(int n)
{
    this->internals->queue = new_teca_data_request_queue(
        this->get_communicator(), n, true, this->get_verbose());
}

// --------------------------------------------------------------------------
unsigned int teca_bayesian_ar_detect::get_thread_pool_size() const noexcept
{
    return this->internals->queue->size();
}

// --------------------------------------------------------------------------
teca_metadata teca_bayesian_ar_detect::get_output_metadata(
    unsigned int port, const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_bayesian_ar_detect::get_output_metadata" << endl;
#endif
    (void)port;

    // if don't already have the parameter table read and distribute it
    if (!this->internals->parameter_table)
    {
        // execute the pipeline that retruns table of parameters
        const_p_teca_dataset parameter_data;

        p_teca_programmable_algorithm capture_parameter_data
            = teca_programmable_algorithm::New();

        capture_parameter_data->set_name("capture_parameter_data");
        capture_parameter_data->set_input_connection(this->internals->parameter_pipeline_port);

        capture_parameter_data->set_execute_callback(
            [&parameter_data] (unsigned int, const std::vector<const_p_teca_dataset> &in_data,
         const teca_metadata &) -> const_p_teca_dataset
         {
             parameter_data = in_data[0];
             return nullptr;
         });

        capture_parameter_data->update();

        int rank = 0;
#if defined(TECA_HAS_MPI)
        MPI_Comm comm = this->get_communicator();
        int is_init = 0;
        MPI_Initialized(&is_init);
        if (is_init)
            MPI_Comm_rank(comm, &rank);
#endif
        // validate the table
        if (rank == 0)
        {
            // did the pipeline run successfully
            const_p_teca_table parameter_table =
                std::dynamic_pointer_cast<const teca_table>(parameter_data);

            if (!parameter_table)
            {
                TECA_ERROR("metadata pipeline failure")
            }
            else if (!parameter_table->has_column(this->min_water_vapor_variable))
            {
                TECA_ERROR("metadata missing percentile column \""
                    << this->min_water_vapor_variable << "\"")
            }
            else if (!parameter_table->get_column(this->min_component_area_variable))
            {
                TECA_ERROR("metadata missing area column \""
                    << this->min_component_area_variable << "\"")
            }
            else if (!parameter_table->get_column(this->hwhm_latitude_variable))
            {
                TECA_ERROR("metadata missing hwhm column \""
                    << this->hwhm_latitude_variable << "\"")
            }
            else
            {
                this->internals->parameter_table = parameter_table;
            }
        }

        // distribute the table to all processes
#if defined(TECA_HAS_MPI)
        if (is_init)
        {
            teca_binary_stream bs;
            if (this->internals->parameter_table && (rank == 0))
                this->internals->parameter_table->to_stream(bs);
            bs.broadcast(comm);
            if (bs && (rank != 0))
            {
               p_teca_table tmp = teca_table::New();
               tmp->from_stream(bs);
               this->internals->parameter_table = tmp;
            }
        }
#endif

        // some already reported error ocurred, bail out here
        if (!this->internals->parameter_table)
            return teca_metadata();

        // check that we have at least one set of parameters
        unsigned long num_params =
            this->internals->parameter_table->get_number_of_rows();

        if (num_params < 1)
        {
            TECA_ERROR("Invalid parameter table, must have at least one row")
            return teca_metadata();
        }
    }

    // this algorithm processes Cartesian mesh based data. It will fetch a
    // timestep and loop over a set of parameters accumulating the result. we
    // report the variable that we compute, for each timestep from the
    // parameter tables.
    teca_metadata md(input_md[0]);
    md.append("variables", std::string("ar_probability"));

    // add attributes to enable CF I/O
    teca_metadata atts;
    md.get("attributes", atts);
    teca_array_attributes prob_atts(
        teca_variant_array_code<float>::get(),
        teca_array_attributes::point_centering,
        0, "unitless", "posterior AR flag",
        "the posterior probability of the presence of an atmospheric river");

    atts.set("ar_probability", (teca_metadata)prob_atts);

    unsigned long num_params =
        this->internals->parameter_table->get_number_of_rows();

    teca_array_attributes count_atts(
        teca_variant_array_code<int>::get(),
        teca_array_attributes::no_centering,
        num_params, "detections", "number of AR detections",
        "number of detections for the parameter table row at the same index in "
        "parameter_table_row");

    atts.set("ar_count", (teca_metadata)count_atts);

    teca_array_attributes row_atts(
        teca_variant_array_code<int>::get(),
        teca_array_attributes::no_centering,
        num_params, "row index", "parameter_table_row",
        "the parameter table row corresponding to the value at the same index "
        "in ar_count");

    atts.set("parameter_table_row", (teca_metadata)row_atts);

    md.set("attributes", atts);

    return md;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_bayesian_ar_detect::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_bayesian_ar_detect::get_upstream_request" << endl;
#endif
    (void) port;
    (void) input_md;

    std::vector<teca_metadata> up_reqs;

    // get the name of the array to request
    if (this->water_vapor_variable.empty())
    {
        TECA_ERROR("A water vapor variable was not specified")
        return up_reqs;
    }

    // pass the incoming request upstream, and
    // add in what we need
    teca_metadata req(request);
    std::set<std::string> arrays;
    if (req.has("arrays"))
        req.get("arrays", arrays);
    arrays.insert(this->water_vapor_variable);

    // remove what we produce
    arrays.erase("ar_probability");
    arrays.erase("ar_count");
    arrays.erase("parameter_table_row");

    req.set("arrays", arrays);

    // send up
    up_reqs.push_back(req);
    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_bayesian_ar_detect::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id() << "teca_bayesian_ar_detect::execute" << endl;
#endif
    (void)port;
    (void)request;

    // check the thread pool
    if (!this->internals->queue)
    {
        TECA_ERROR("thread pool has not been created. Did you forget "
            "to call set_thread_pool_size?")
        return nullptr;
    }

    // check the parameter table
    if (!this->internals->parameter_table)
    {
        TECA_ERROR("empty parameter table input")
        return nullptr;
    }

    // get the input
    p_teca_dataset in_data =
        std::const_pointer_cast<teca_dataset>(input_data[0]);

    p_teca_cartesian_mesh in_mesh =
        std::dynamic_pointer_cast<teca_cartesian_mesh>(in_data);
    if (!in_mesh)
    {
        TECA_ERROR("empty mesh input, or not a cartesian_mesh")
        return nullptr;
    }

    const teca_metadata &in_md = in_mesh->get_metadata();

    std::string index_request_key;
    if (in_md.get("index_request_key", index_request_key))
    {
        TECA_ERROR("Dataset metadata is missing the index_request_key key")
        return nullptr;
    }

    unsigned long index = 0;
    if (in_md.get(index_request_key, index))
    {
        TECA_ERROR("Dataset metadata is missing the \""
            << index_request_key << "\" key")
        return nullptr;
    }

    double time = 0.0;
    if (in_mesh->get_time(time) &&
        request.get("time", time))
    {
        TECA_ERROR("request missing \"time\"")
        return nullptr;
    }

    // build the parameter table reduction pipeline
    unsigned long parameter_table_size =
        this->internals->parameter_table->get_number_of_rows();

    // the executive will loop over table rows, the top of the pipeline
    // will ignore the incoming request which is for a specific table row
    // and always pass the input mesh down stream
    ::parameter_table_request_generator request_gen(parameter_table_size,
            this->internals->parameter_table->get_column(this->hwhm_latitude_variable),
            this->internals->parameter_table->get_column(this->min_water_vapor_variable),
            this->internals->parameter_table->get_column(this->min_component_area_variable));

    teca_metadata exec_md;
    request_gen.initialize_index_executive(exec_md);

    // set up the pipeline source
    p_teca_programmable_algorithm dss = teca_programmable_algorithm::New();
    dss->set_name("dataset_source");
    dss->set_communicator(MPI_COMM_SELF);
    dss->set_number_of_input_connections(0);
    dss->set_number_of_output_ports(1);
    dss->set_report_callback(
        [&exec_md](unsigned int, const std::vector<teca_metadata>&) -> teca_metadata
        {
            return exec_md;
        });
    dss->set_execute_callback(
        [& in_mesh] (unsigned int, const std::vector<const_p_teca_dataset> &,
     const teca_metadata &req) -> const_p_teca_dataset
     {
         int row_id = 0;
         if (req.get("row_id", row_id))
         {
             TECA_ERROR("failed to get parameter table row")
             return nullptr;
         }

         p_teca_dataset out_mesh = in_mesh->new_instance();
         out_mesh->shallow_copy(in_mesh);
         out_mesh->get_metadata().set("parameter_table_row", row_id);

         return out_mesh;
     });

    // set up the filtering stages. these extract control parameters
    // which are served up from the parameter table from the incoming request
    p_teca_latitude_damper damp = teca_latitude_damper::New();
    damp->set_communicator(MPI_COMM_SELF);
    damp->set_input_connection(dss->get_output_port());
    damp->set_damped_variables({this->water_vapor_variable});

    p_teca_binary_segmentation seg = teca_binary_segmentation::New();
    seg->set_communicator(MPI_COMM_SELF);
    seg->set_input_connection(damp->get_output_port());
    seg->set_threshold_variable(this->water_vapor_variable);
    seg->set_segmentation_variable("wv_seg");
    seg->set_threshold_by_percentile();

    p_teca_connected_components cc = teca_connected_components::New();
    cc->set_communicator(MPI_COMM_SELF);
    cc->set_input_connection(seg->get_output_port());
    cc->set_segmentation_variable("wv_seg");
    cc->set_component_variable("wv_cc");

    p_teca_2d_component_area ca = teca_2d_component_area::New();
    ca->set_communicator(MPI_COMM_SELF);
    ca->set_input_connection(cc->get_output_port());
    ca->set_component_variable("wv_cc");
    ca->set_contiguous_component_ids(1);

    p_teca_component_area_filter caf = teca_component_area_filter::New();
    caf->set_communicator(MPI_COMM_SELF);
    caf->set_input_connection(ca->get_output_port());
    caf->set_component_variable("wv_cc");
    caf->set_contiguous_component_ids(1);

    // set up the request generator. 1 request per parameter table row is
    // generated. the request is populated with values in columns of that row
    p_teca_programmable_algorithm pa = teca_programmable_algorithm::New();
    pa->set_name("request_generator");
    pa->set_communicator(MPI_COMM_SELF);
    pa->set_number_of_input_connections(1);
    pa->set_number_of_output_ports(1);
    pa->set_input_connection(caf->get_output_port());
    pa->set_request_callback(request_gen);

    // set up the reduction which computes the average over runs of all control
    // parameter combinations provided in the parameter table
    ::parameter_table_reduction reduce(parameter_table_size,
        "wv_cc", "ar_probability");

    p_teca_programmable_reduce pr = teca_programmable_reduce::New();
    pr->set_name("parameter_table_reduce");
    pr->set_communicator(MPI_COMM_SELF);
    pr->set_input_connection(pa->get_output_port());
    pr->set_reduce_callback(reduce);
    pr->set_finalize_callback(reduce);
    pr->set_stream_size(2);
    pr->set_verbose(0);
    pr->set_data_request_queue(this->internals->queue);

    // extract the result
    p_teca_dataset_capture dc = teca_dataset_capture::New();
    dc->set_communicator(MPI_COMM_SELF);
    dc->set_input_connection(pr->get_output_port());
    dc->set_executive(teca_index_executive::New());

    // run the pipeline
    dc->update();

    // get the pipeline output and normalize the probabilty field
    p_teca_cartesian_mesh out_mesh =
        std::dynamic_pointer_cast<teca_cartesian_mesh>(
            std::const_pointer_cast<teca_dataset>(dc->get_dataset()));

    if (!out_mesh)
    {
        TECA_ERROR("Pipeline execution failed")
        return nullptr;
    }

    // reset the pipeline control keys
    teca_metadata &out_md = out_mesh->get_metadata();
    out_md.set("index_request_key", index_request_key);
    out_md.set(index_request_key, index);
    out_md.set("time", time);

    return out_mesh;
}
