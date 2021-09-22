#include "teca_cf_time_axis_reader.h"

#include "teca_mpi.h"
#include "teca_config.h"
#include "teca_cf_time_axis_data.h"
#include "teca_file_util.h"
#include "teca_netcdf_util.h"



// --------------------------------------------------------------------------
teca_cf_time_axis_reader::teca_cf_time_axis_reader() : t_axis_variable("time")
{
}

// --------------------------------------------------------------------------
void teca_cf_time_axis_reader::set_modified()
{
    // clear cached metadata before forwarding on to
    // the base class.
    this->files.clear();
    this->path.clear();
    teca_algorithm::set_modified();
}

// --------------------------------------------------------------------------
teca_metadata teca_cf_time_axis_reader::get_output_metadata(unsigned int,
    const std::vector<teca_metadata> &)
{
    int rank = 0;
#if defined(TECA_HAS_MPI)
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
    {
        MPI_Comm comm = this->get_communicator();

        // this rank has been excluded from execution
        if (comm == MPI_COMM_NULL)
            return teca_metadata();

        MPI_Comm_rank(comm, &rank);
    }
#endif

    if (this->files.empty())
    {
        teca_binary_stream stream;

        if (rank == 0)
        {
            std::string tmp_path;
            std::vector<std::string> tmp_files;

            if (!this->file_names.empty())
            {
                // use a list of file names
                tmp_path = teca_file_util::path(this->file_names[0]);
                size_t n_file_names = this->file_names.size();
                for (size_t i = 0; i < n_file_names; ++i)
                {
                    tmp_files.push_back(teca_file_util::filename(file_names[i]));
                }
            }
            else
            {
                // use a regex
                std::string regex = teca_file_util::filename(this->files_regex);
                tmp_path = teca_file_util::path(this->files_regex);

                if (teca_file_util::locate_files(tmp_path, regex, tmp_files))
                {
                    TECA_FATAL_ERROR(
                        << "Failed to locate any files" << std::endl
                        << this->files_regex << std::endl
                        << tmp_path << std::endl
                        << regex)
                }
            }

            // package the list of files. these are duplicated on all ranks
            // because the read will occur in parallel and the excutive will
            // decide which rank reads which files
            stream.pack(tmp_path);
            stream.pack(tmp_files);
        }

        stream.broadcast(this->get_communicator());
        stream.unpack(this->path);
        stream.unpack(this->files);
        stream.clear();

        if (this->files.size() < 1)
            return teca_metadata();
    }

    teca_metadata md_out;
    md_out.set("index_initializer_key", std::string("number_of_files"));
    md_out.set("number_of_files", this->files.size());
    md_out.set("index_request_key", std::string("file_id"));

    return md_out;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_cf_time_axis_reader::execute(unsigned int,
    const std::vector<const_p_teca_dataset> &,
    const teca_metadata &request)
{
#if defined(TECA_HAS_MPI)
    MPI_Comm comm = this->get_communicator();

    // this rank was excluded from execution
    if (comm == MPI_COMM_NULL)
        return nullptr;
#endif

    unsigned long file_id = 0;
    if (request.get("file_id", file_id))
    {
        TECA_FATAL_ERROR("Invalid file_id " << file_id)
        return nullptr;
    }

    teca_netcdf_util::read_variable_and_attributes
        read(this->path, this->files[file_id], file_id, this->t_axis_variable);

    teca_netcdf_util::read_variable_and_attributes::data_t axis_data = read();

    p_teca_cf_time_axis_data data_out = teca_cf_time_axis_data::New();
    data_out->transfer(file_id, std::move(axis_data.second));

    return data_out;
}
