#include "teca_spatial_reduction.h"

#include "teca_cartesian_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_metadata.h"
#include "teca_calendar_util.h"
#include "teca_valid_value_mask.h"
#include "teca_table.h"
#include "teca_coordinate_util.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <cmath>
#include <typeinfo>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

#define TECA_DEBUG 0

using namespace teca_variant_array_util;
using allocator = teca_variant_array::allocator;

// PIMPL idiom hides internals
// defines the API for reduction operators
class teca_spatial_reduction::internals_t
{
public:
    internals_t() {}
    ~internals_t() {}

    /** check if the passed array contains integer data, if so deep-copy to
     * floating point type. 32(64) bit integers will be copied to 32(64) bit
     * floating point.
     * @param[in] alloc the allocator to use for the new array if a deep-copy is made
     * @param[inout] array the array to check and convert from integer to floating point
     */
    static
    const_p_teca_variant_array
    ensure_floating_point(allocator alloc, const const_p_teca_variant_array &array);

public:
    class reduction_operator;
    class average_operator;
    class reduction_operator_factory;

    using p_reduction_operator = std::shared_ptr<reduction_operator>;

    void set_operation(const std::string &array, const p_reduction_operator &op);
    p_reduction_operator &get_operation(const std::string &array);

public:
    std::mutex m_mutex;
    teca_metadata metadata;
    std::map<std::thread::id, std::map<std::string, p_reduction_operator>> operation;
};

// --------------------------------------------------------------------------
void teca_spatial_reduction::internals_t::set_operation(
    const std::string &array, const p_reduction_operator &op)
{
    auto tid = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(m_mutex);
    this->operation[tid][array] = op;
}

// --------------------------------------------------------------------------
teca_spatial_reduction::internals_t::p_reduction_operator &
teca_spatial_reduction::internals_t::get_operation(const std::string &array)
{
    auto tid = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(m_mutex);
    return this->operation[tid][array];
}

// --------------------------------------------------------------------------
const_p_teca_variant_array
teca_spatial_reduction::internals_t::ensure_floating_point(
     allocator alloc, const const_p_teca_variant_array &array)
{
    if (std::dynamic_pointer_cast<const teca_variant_array_impl<double>>(array)  ||
        std::dynamic_pointer_cast<const teca_variant_array_impl<float>>(array))
    {
        // the data is already floating point type
        return array;
    }
    else if (std::dynamic_pointer_cast<const teca_variant_array_impl<long>>(array) ||
        std::dynamic_pointer_cast<const teca_variant_array_impl<long long>>(array) ||
        std::dynamic_pointer_cast<const teca_variant_array_impl<unsigned long>>(array) ||
        std::dynamic_pointer_cast<const teca_variant_array_impl<unsigned long long>>(array))
    {
        // convert from a 64 bit integer to a 64 bit floating point
        size_t n_elem = array->size();
        p_teca_double_array tmp = teca_double_array::New(n_elem, alloc);
        tmp->set(0, array, 0, n_elem);
        return tmp;
    }
    else
    {
        // convert from a 32 bit integer to a 32 bit floating point
        size_t n_elem = array->size();
        p_teca_float_array tmp = teca_float_array::New(n_elem, alloc);
        tmp->set(0, array, 0, n_elem);
        return tmp;
    }
}

class teca_spatial_reduction::internals_t::reduction_operator
{
public:
    virtual ~reduction_operator() {}

    reduction_operator() : fill_value(-1), result(nullptr), valid(nullptr),
                           land_weights (nullptr), land_weights_norm(1) {}

    virtual void initialize(double fill_value);

    virtual int update_cpu(int device_id,
          const const_p_teca_variant_array &array,
          const const_p_teca_variant_array &valid,
          unsigned long timesteps_per_request) = 0;


public:
    double fill_value;
    p_teca_variant_array result;
    p_teca_variant_array valid;
    const_p_teca_variant_array land_weights;
    double land_weights_norm;
};


/// implements 2D spatial average
class teca_spatial_reduction::internals_t::average_operator :
      public teca_spatial_reduction::internals_t::reduction_operator
{
public:
    int update_cpu(int device_id,
          const const_p_teca_variant_array &array,
          const const_p_teca_variant_array &valid,
          unsigned long timesteps_per_request) override;
};

/// constructs reduction_operator
class teca_spatial_reduction::internals_t::reduction_operator_factory
{
public:
    /** Allocate and return an instance of the named operator
     * @param[in] op Id of the desired reduction operator.
     *               One of average, summation, minimum, or
     *                                              maximum
     * @returns an instance of reduction_operator
     */
    static teca_spatial_reduction::internals_t::p_reduction_operator New(
                                                                   int op);
};

// --------------------------------------------------------------------------
void teca_spatial_reduction::internals_t::reduction_operator::initialize(
     double fill_value)
{
    this->fill_value = fill_value;
}

// --------------------------------------------------------------------------
int teca_spatial_reduction::internals_t::average_operator::update_cpu(
    int device_id,
    const const_p_teca_variant_array &input_array,
    const const_p_teca_variant_array &in_valid,
    unsigned long timesteps_per_request)
{
    (void)device_id;

    // don't use integer types for this calculation
    allocator alloc = allocator::malloc;
    auto in_array = internals_t::ensure_floating_point(alloc, input_array);

    unsigned long n_elem = in_array->size();
    unsigned long n_elem_per_timestep = n_elem/timesteps_per_request;

    VARIANT_ARRAY_DISPATCH(in_array.get(),

       NT weights_norm = NT(this->land_weights_norm);
       NT fill_value = NT(this->fill_value);

       auto [sp_in_array, p_in_array] = get_host_accessible<CTT>(in_array);
       auto [sp_weights, p_weights] = get_host_accessible<CTT>(this->land_weights);

       this->result = teca_variant_array_impl<NT>::New(timesteps_per_request, 0.);
       auto [p_res_array] = data<TT>(this->result);

       if (in_valid)
       {
          auto [sp_in_valid, p_in_valid] = get_host_accessible<CTT_MASK>(in_valid);

          this->valid = teca_variant_array_impl<NT_MASK>::New(timesteps_per_request, NT_MASK(0));
          auto [p_res_valid] = data<TT_MASK>(this->valid);

          for (unsigned int i = 0; i < timesteps_per_request; ++i)
          {
              for (unsigned int j = 0; j < n_elem_per_timestep; ++j)
              {
                  if (p_in_valid[j+i*n_elem_per_timestep])
                  {
                      p_res_array[i] += p_in_array[j+i*n_elem_per_timestep] * p_weights[j];
                      p_res_valid[i] = NT_MASK(1);
                  }
              }
              p_res_array[i] = (!p_res_valid[i]) ?
                               fill_value : p_res_array[i] / weights_norm;
          }
       }
       else
       {
          // update, no missing data
          for (unsigned int i = 0; i < timesteps_per_request; ++i)
          {
              for (unsigned int j = 0; j < n_elem_per_timestep; ++j)
              {
                  p_res_array[i] += p_in_array[j+i*n_elem_per_timestep] * p_weights[j];
              }
              p_res_array[i] = p_res_array[i] / weights_norm;
          }
       }
    )

    return 0;
}

// --------------------------------------------------------------------------
teca_spatial_reduction::internals_t::p_reduction_operator
    teca_spatial_reduction::internals_t::reduction_operator_factory::New(
    int op)
{
    if (op == average)
    {
        return std::make_shared<average_operator>();
    }
    TECA_ERROR("Failed to construct a \""
        << op << "\" reduction operator")
    return nullptr;
}

// --------------------------------------------------------------------------
int teca_spatial_reduction::set_operation(const std::string &op)
{
    if (op == "average")
    {
        this->operation = average;
    }
    else
    {
        TECA_FATAL_ERROR("Invalid operator name \"" << op << "\"")
        return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
std::string teca_spatial_reduction::get_operation_name()
{
    std::string name;
    switch(this->operation)
    {
        case average:
            name = "average";
            break;
        default:
            TECA_FATAL_ERROR("Invalid \"operator\" " << this->operation)
    }
    return name;
}

// --------------------------------------------------------------------------
teca_spatial_reduction::teca_spatial_reduction() :
    operation(average), fill_value(-1), land_weights(""),
    land_weights_norm(1)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);

    this->set_stream_size(1);

    this->internals = new teca_spatial_reduction::internals_t;
}

// --------------------------------------------------------------------------
teca_spatial_reduction::~teca_spatial_reduction()
{
    delete this->internals;
}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_spatial_reduction::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_spatial_reduction":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::vector<std::string>, prefix, point_arrays,
            "list of point centered arrays to process")
        TECA_POPTS_GET(int, prefix, operation,
            "reduction operator to use"
            " (average)")
        TECA_POPTS_GET(double, prefix, fill_value,
            "the value of the NetCDF _FillValue attribute")
        TECA_POPTS_GET(std::string, prefix, land_weights,
            "")
        TECA_POPTS_GET(double, prefix, land_weights_norm,
            "")
        ;

    this->teca_threaded_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_spatial_reduction::set_properties(
    const std::string &prefix, variables_map &opts)
{
    this->teca_threaded_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, std::vector<std::string>, prefix, point_arrays)
    TECA_POPTS_SET(opts, int, prefix, operation)
    TECA_POPTS_SET(opts, double, prefix, fill_value)
    TECA_POPTS_SET(opts, std::string, prefix, land_weights)
    TECA_POPTS_SET(opts, double, prefix, land_weights_norm)
}
#endif

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_spatial_reduction::get_upstream_request(
    unsigned int port,
    const std::vector<teca_metadata> &md_in,
    const teca_metadata &req_in)
{
    (void)port;

    if (this->get_verbose() > 0)
    {
        std::cerr << teca_parallel_id()
            << "teca_spatial_reduction::get_upstream_request" << std::endl;
    }

    const teca_metadata md = md_in[0];

    // get the available arrays
    std::set<std::string> vars_in;
    if (md.has("variables"))
       md.get("variables", vars_in);

    // get the array attributes
    teca_metadata atts;
    md.get("attributes", atts);

    // get the requested arrays
    std::set<std::string> arrays;
    if (req_in.has("arrays"))
        req_in.get("arrays", arrays);

    size_t n_array = this->point_arrays.size();
    for (size_t i = 0; i < n_array; ++i)
    {
        const std::string &array = this->point_arrays[i];

        // request the array
        if (!arrays.count(array))
            arrays.insert(array);

        double fill_value = -1;
        std::string vv_mask = array + "_valid";
        if (vars_in.count(vv_mask) && !arrays.count(vv_mask))
        {
            // request the associated valid value mask
            arrays.insert(vv_mask);

            // get the fill value
            teca_metadata array_atts;
            atts.get(array, array_atts);
            if (this->fill_value != -1)
            {
                fill_value = this->fill_value;
            }
            else if (array_atts.has("_FillValue"))
            {
                array_atts.get("_FillValue", fill_value);
            }
            else if (array_atts.has("missing_value"))
            {
                array_atts.get("missing_value", fill_value);
            }
        }

        // create and initialize the operator
        internals_t::p_reduction_operator op
             = internals_t::reduction_operator_factory::New(this->operation);

        op->initialize(fill_value);

        // save the operator
        this->internals->set_operation(array, op);
    }

    if (!arrays.count(this->land_weights))
       arrays.insert(this->land_weights);

    teca_metadata req(req_in);
    req.set("arrays", arrays);
    std::vector<teca_metadata> up_reqs;
    up_reqs.push_back(req);

    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_spatial_reduction::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &data_in,
    const teca_metadata &req_in,
    int streaming)
{
    (void)port;
    (void)req_in;
    (void)streaming;

    // get the assigned GPU or CPU
    int device_id = -1;

    if (this->get_verbose() > 0)
    {
        std::cerr << teca_parallel_id()
            << "teca_spatial_reduction::execute"
            << " device " << device_id
            << " " << data_in.size() << std::endl;
    }

    // get the incoming mesh
    auto mesh_in = std::dynamic_pointer_cast<const teca_cartesian_mesh>(data_in[0]);
    auto const& arrays_in = mesh_in->get_point_arrays();

    auto land_weights = arrays_in->get(this->land_weights);
    if (!land_weights)
    {
        TECA_FATAL_ERROR("array \"" << this->land_weights << "\" not found")
        return nullptr;
    }

    unsigned long t_ext[2] = {0ul};
    mesh_in->get_temporal_extent(t_ext);
    unsigned long timesteps_per_request = t_ext[1] - t_ext[0] + 1;

    p_teca_table out_table = teca_table::New();

    // get offset units
    std::string time_units;
    mesh_in->get_time_units(time_units);
    out_table->set_time_units(time_units);

    // get offset calendar
    std::string calendar;
    mesh_in->get_calendar(calendar);
    out_table->set_calendar(calendar);

    unsigned long time_step;
    mesh_in->get_time_step(time_step);
    double time_offset = 0.0;
    mesh_in->get_time(time_offset);
    out_table->declare_columns("step", long(), "time", double());
    out_table << time_step << time_offset;

    size_t n_array = this->point_arrays.size();
    for (size_t j = 0; j < n_array; ++j)
    {
        const std::string &array = this->point_arrays[j];

        // get the incoming data array
        auto array_in = arrays_in->get(array);
        if (!array_in)
        {
            TECA_FATAL_ERROR("array \"" << array << "\" not found")
            return nullptr;
        }

        // get the incoming valid value mask
        std::string valid = array + "_valid";
        auto valid_in = arrays_in->get(valid);

        // apply the reduction
        auto &op = this->internals->get_operation(array);

        op->land_weights = land_weights;
        op->land_weights_norm = this->land_weights_norm;
        op->update_cpu(device_id, array_in, valid_in, timesteps_per_request);

        out_table->append_column(array, op->result);
    }

#if TECA_DEBUG > 1
    out_table->to_stream(cerr);
    std::cerr << std::endl;
#endif

    return out_table;
}
