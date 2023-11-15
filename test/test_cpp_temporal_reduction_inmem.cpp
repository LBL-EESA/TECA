#include "teca_config.h"
#include "teca_algorithm.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_array_attributes.h"
#include "teca_metadata.h"
#include "teca_cartesian_mesh.h"
#include "teca_cartesian_mesh_source.h"
#include "teca_temporal_reduction.h"
#include "teca_cartesian_mesh_regrid.h"
#include "teca_cf_writer.h"
#include "teca_mpi_manager.h"
#include "teca_system_interface.h"

#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>

#if defined(TECA_HAS_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <hamr_buffer.h>
#include <hamr_buffer_allocator.h>
#include <hamr_buffer_transfer.h>

using namespace teca_variant_array_util;
using allocator = teca_variant_array::allocator;


/// get the number of seconds per time unit
int get_number_of_seconds(const std::string &unit, double &sec)
{
    if (unit.find("seconds") != std::string::npos)
    {
        sec = 1.0;
    }
    else if (unit.find("minutes") != std::string::npos)
    {
        sec = 60.0;
    }
    else if (unit.find("hours") != std::string::npos)
    {
        sec = 60.0*60.0;
    }
    else if (unit.find("days") != std::string::npos)
    {
        sec = 60.0*60.0*24.0;
    }
    else
    {
        sec = 0.0;
        TECA_ERROR("Unsupported time axis units " << unit)
        return -1;
    }
    return 0;
}

#if defined(TECA_HAS_CUDA)
namespace cuda_impl
{


__global__
void generate(double *f_xyzt, size_t nxyz, size_t nt, double t0, double dt,
              double *f, double *a, double b, size_t nf)
{
    double pi = 3.14159235658979;

    size_t q = blockIdx.x*blockDim.x + threadIdx.x;
    if (q >= nxyz)
        return;

    for (size_t i = 0; i < nt; ++i)
    {
        double f_t = 0.0;
        double t = t0 + i * dt;
        for (size_t j = 0; j < nf; ++j)
        {
            f_t += a[j] * sin( 2*pi*f[j]*t );
        }
        f_xyzt[i*nxyz + q] = f_t + b;
    }
}

// **************************************************************************
int
generate(int device, size_t nx, size_t ny, size_t nz, size_t nt,
    double t0, double dt, const std::vector<double> &frequencies,
    const std::vector<double> &amplitudes, double bias,
    p_teca_double_array &f_t)
{
    cudaError_t ierr = cudaSuccess;

    cudaSetDevice(device);
    cudaStream_t strm cudaStreamPerThread;

    size_t nf = frequencies.size();

    // move the frequencies and amplitudes to the device.
    hamr::buffer<double> freqs(hamr::buffer_allocator::cuda_async, strm,
                               hamr::buffer_transfer::async, nf, frequencies.data());

    hamr::buffer<double> amps(hamr::buffer_allocator::cuda_async, strm,
                              hamr::buffer_transfer::async, nf, amplitudes.data());

    // allocate space for the result
    size_t nxyz = nx*ny*nz;
    size_t nxyzt = nxyz*nt;

    double *pf_t;
    std::tie( f_t, pf_t ) = ::New<teca_double_array>(nxyzt, allocator::cuda_async);

    int n_threads = 128;
    int n_blocks = nxyz / n_threads + ( nxyz % n_threads ? 1 : 0 );

    generate<<<n_blocks, n_threads, 0, strm>>>(pf_t, nxyz, nt, t0, dt,
                                               freqs.data(), amps.data(), bias, nf);

    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        TECA_ERROR("Failed to generate data. " << cudaGetErrorString(ierr))
        return -1;
    }

    return 0;
}
}
#endif

namespace cpu_impl
{
// **************************************************************************
int
generate(size_t nx, size_t ny, size_t nz, size_t nt,
    double t0, double dt, const std::vector<double> &f,
    const std::vector<double> &a, double b,
    p_teca_double_array &f_xyzt)
{
    // allocate space for the result
    size_t nxyz = nx*ny*nz;
    size_t nxyzt = nxyz*nt;

    double *pf_xyzt;
    std::tie(f_xyzt, pf_xyzt) = ::New<teca_double_array>(nxyzt, allocator::malloc);

    double pi = 3.14159235658979;

    size_t nf = f.size();

    for (size_t i = 0; i < nt; ++i)
    {
        double f_t = 0.0;
        double t = t0 + i * dt;
        for (size_t j = 0; j < nf; ++j)
        {
            f_t += a[j] * sin( 2*pi*f[j]*t );
        }

        f_t += b;

        double *pf_xyzt_i = pf_xyzt + i*nxyz;
        for (size_t q = 0; q < nxyz; ++q)
        {
            pf_xyzt_i[q] = f_t;
        }
    }

    return 0;
}
}


TECA_SHARED_OBJECT_FORWARD_DECL(generate_time_series)

/**
 * This class generates 4D point centered data according to the function:
 *
 *   f(x,y,z,t) = sin( 2*pi*f1*t ) + sin( 2*pi*f2*t ) + ... sin ( 2*pi*fn*t )
 *
 * The values of f1, f2, ... fn are in units of Hz
 */
class generate_time_series : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(generate_time_series)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(generate_time_series)
    TECA_ALGORITHM_CLASS_NAME(generate_time_series)
    ~generate_time_series() override {}

    /// set the bias of the sin waves
    void set_bias(double bias) { m_bias = bias; }

    /// set the list of amplitudes of the sin waves
    void set_amplitudes(std::vector<double> &amps) { m_amplitudes = amps; }

    /// set the list of frequencies of the sin waves
    void set_frequencies(std::vector<double> &freqs) { m_frequencies = freqs; }

    /// get the names of the arrays this calss generates
    std::vector<std::string> get_point_array_names() { return {"f_t"}; }

protected:
    generate_time_series() : m_verbose(0), m_frequencies{},
                             m_amplitudes{}, m_bias(0.0)
    {
        this->set_number_of_input_connections(1);
        this->set_number_of_output_ports(1);
    }

    teca_metadata get_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    int m_verbose;
    std::vector<double> m_frequencies;
    std::vector<double> m_amplitudes;
    double m_bias;

};


// --------------------------------------------------------------------------
teca_metadata generate_time_series::get_output_metadata(unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
    (void) port;

    // report arrays we generate
    auto md_out = teca_metadata(input_md[0]);

    std::vector<std::string> arrays;
    md_out.get("arrays", arrays);
    arrays.push_back("f_t");
    md_out.set("arrays", arrays);

    // get the extent of the dataset
    unsigned long wext[6] = {0ul};
    md_out.get("whole_extent", wext);

    unsigned long ncells = (wext[1] - wext[0] + 1) *
             (wext[3] - wext[2] + 1) * (wext[5] - wext[4] + 1);

    // create the metadata for the writer
    auto faa = teca_array_attributes(
        teca_variant_array_code<double>::get(),
        teca_array_attributes::point_centering,
        ncells, {1,1,1,1}, "none", "f(t)",
        "function of time");

    // put it in the array attributes
    teca_metadata atts;
    md_out.get("attributes", atts);

    atts.set("f_t", (teca_metadata)faa);
    md_out.set("attributes", atts);

    return md_out;
}

// --------------------------------------------------------------------------
const_p_teca_dataset generate_time_series::execute(unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
    (void) port;
    (void) request;

    // get the requested target device
    int device_id = -1;
#if defined(TECA_HAS_CUDA)
    request.get("device_id", device_id);
    if (device_id >= 0)
    {
        cudaSetDevice(device_id);
    }
#endif

    // get the input
    auto mesh_in = std::dynamic_pointer_cast
        <const teca_cartesian_mesh>(input_data[0]);

    // get the time units in seconds
    std::string t_units;
    mesh_in->get_time_units(t_units);

    double seconds_per = 0.0;
    get_number_of_seconds(t_units, seconds_per);

    // get mesh dims and coordinate arrays
    unsigned long ext[6] = {0lu};
    mesh_in->get_extent(ext);
    unsigned long nx = ext[1] - ext[0] + 1;
    unsigned long ny = ext[3] - ext[2] + 1;
    unsigned long nz = ext[5] - ext[4] + 1;

    // get the time step size
    unsigned long text[2] = {0lu};
    mesh_in->get_temporal_extent(text);
    unsigned long nt = text[1] - text[0] + 1;

    double t_bds[2] = {0.0};
    mesh_in->get_temporal_bounds(t_bds);
    double dt = seconds_per * ( t_bds[1] - t_bds[0] );
    dt = nt > 1 ? dt / ( nt - 1 ) : dt;
    double t0 = t_bds[0] * seconds_per;

    // generate the output
    int ierr = 0;
    p_teca_double_array f_t;
#if defined(TECA_HAS_CUDA)
    if (device_id >= 0)
    {
        ierr = cuda_impl::generate(device_id, nx, ny, nz, nt,
                                   t0, dt, m_frequencies, m_amplitudes, m_bias,
                                   f_t);
    }
    else
    {
#endif
        ierr = cpu_impl::generate(nx, ny, nz, nt, t0, dt,
                                  m_frequencies, m_amplitudes, m_bias,
                                  f_t);

#if defined(TECA_HAS_CUDA)
    }
#endif
    if (ierr)
    {
        TECA_FATAL_ERROR("Failed to generate data on device("
            << device_id << ")")
        return nullptr;
    }

    // create the output and add in the arrays
    auto mesh_out = teca_cartesian_mesh::New();
    mesh_out->shallow_copy(std::const_pointer_cast<teca_cartesian_mesh>(mesh_in));
    mesh_out->get_point_arrays()->append("f_t", f_t);

    if (this->get_verbose())
    {
        TECA_STATUS("Generated time series for steps "
            << text << " on device(" << device_id << ")")
    }

    return mesh_out;
}




int main(int argc, char **argv)
{
    teca_mpi_manager mpi_man(argc, argv);
    int rank = mpi_man.get_comm_rank();
    int n_ranks = mpi_man.get_comm_size();

    teca_system_interface::set_stack_trace_on_error();
    teca_system_interface::set_stack_trace_on_mpi_error();

    if (argc != 17)
    {
        std::cerr << "test_temporal_reduction [nx in] [ny in] [nz in]"
            " [steps per day] [num years]"
            " [num reduction threads] [threads per device]"
            " [reduction interval] [reduction operator] [steps per request]"
            " [out file] [file layout] [num writer threads]"
            " [nx out] [ny out] [nz out]" << std::endl;
        return -1;
    }

    unsigned long nx_in = atoi(argv[1]);
    unsigned long ny_in = atoi(argv[2]);
    unsigned long nz_in = atoi(argv[3]);

    double steps_per_day = atof(argv[4]);
    double n_years = atof(argv[5]);
    //double dt = 1.0 / steps_per_day;
    unsigned long nt = n_years * 360.0 * steps_per_day;

    int n_red_threads = atoi(argv[6]);
    int threads_per_dev = atoi(argv[7]);
    std::string red_int = argv[8];
    std::string red_op = argv[9];
    int steps_per_req = atoi(argv[10]);

    std::string ofile_name = argv[11];
    std::string layout = argv[12];
    int n_wri_threads = atoi(argv[13]);

    unsigned long nx_out = atoi(argv[14]);
    unsigned long ny_out = atoi(argv[15]);
    unsigned long nz_out = atoi(argv[16]);

    double T0 = 360.0; // period of 1 year
    double T1 = 1.0;   // period of 1 day
    double seconds_per_day = 60.0*60.0*24.0;
    double f0 = 1.0 / ( T0 * seconds_per_day );
    double f1 = 1.0 / ( T1 * seconds_per_day );
    std::vector<double> freq{f0, f1};
    std::vector<double> amp{35.0, 15.0};
    double bias = 0.0;

    if (rank == 0)
    std::cerr << "n_ranks=" << n_ranks
        << "  nx_in=" << nx_in << "  ny_in=" << ny_in << "  nz_in=" << nz_in
        << "  nt=" << nt << "  reduce_threads=" << n_red_threads
        << "  threads_per_dev=" << threads_per_dev
        << "  red_int=" << red_int << "  red_op=" << red_op
        << "  steps_per_req=" << steps_per_req
        << "  layout=" << layout << "  writer_threads=" << n_wri_threads
        << "  nx_out=" << nx_out << "  ny_out=" << ny_out << "  nz_out=" << nz_out
        << std::endl;

    // create the high res input mesh
    auto ts_mesh = teca_cartesian_mesh_source::New();
    ts_mesh->set_calendar("360_day", "days since 1980-01-01 00:00:00");;
    ts_mesh->set_whole_extents({0, nx_in - 1, 0, ny_in - 1, 0, nz_in - 1, 0, nt - 1});
    ts_mesh->set_bounds({0.0, 360.0, -90.0, 90.0, 92500.0, 5000.0, 0.0, n_years*360.0});

    // generate time series data
    auto ts_gen = generate_time_series::New();
    ts_gen->set_input_connection(ts_mesh->get_output_port());
    ts_gen->set_amplitudes(amp);
    ts_gen->set_frequencies(freq);
    ts_gen->set_bias(bias);

    // temporal reduction
    auto reduc = teca_cpp_temporal_reduction::New();
    reduc->set_input_connection(ts_gen->get_output_port());
    reduc->set_verbose(1);
    reduc->set_threads_per_device(threads_per_dev);
    reduc->set_thread_pool_size(n_red_threads);
    reduc->set_interval(red_int);
    reduc->set_operation(red_op);
    reduc->set_point_arrays({"f_t"});
    reduc->set_steps_per_request(steps_per_req);

    // low res output mesh
    auto md = reduc->update_metadata();
    auto io_mesh = teca_cartesian_mesh_source::New();
    io_mesh->set_calendar("360_day", "days since 1980-01-01 00:00:00");;
    io_mesh->set_whole_extents({0, 0, 0, 0, 0, 0, 0, 0});
    io_mesh->set_bounds({0.0, 360.0, -90.0, 90.0, 92500.0, 5000.0, 0.0, 0.0});
    io_mesh->set_t_axis(md);

    // remesh stage
    auto regrid = teca_cartesian_mesh_regrid::New();
    regrid->set_input_connection(0, io_mesh->get_output_port());
    regrid->set_input_connection(1, reduc->get_output_port());
    regrid->set_interpolation_mode_nearest();

    // writer
    auto cfw = teca_cf_writer::New();
    cfw->set_input_connection(regrid->get_output_port());
    cfw->set_verbose(1);
    cfw->set_thread_pool_size(n_wri_threads);
    cfw->set_file_name(ofile_name);
    cfw->set_layout(layout);
    cfw->set_point_arrays({"f_t"});

    cfw->update();

    return 0;
}
