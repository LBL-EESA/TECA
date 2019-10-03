#include "teca_cartesian_mesh_writer.h"

#include "teca_config.h"
#include "teca_cartesian_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_file_util.h"
#include "teca_vtk_util.h"


#if defined(TECA_HAS_VTK) || defined(TECA_HAS_PARAVIEW)
#include "vtkRectilinearGrid.h"
#include "vtkXMLRectilinearGridWriter.h"
#endif

#include <iostream>
#include <sstream>
#include <cstring>
#include <cerrno>
#include <string>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif
#if defined(TECA_HAS_MPI)
#include <mpi.h>
#endif

namespace internals {

void write_vtk_array_data(FILE *ofile,
    const const_p_teca_variant_array &a, int binary)
{
    if (a)
    {
        size_t na = a->size();
        TEMPLATE_DISPATCH(const teca_variant_array_impl,
            a.get(),
            const NT *pa = dynamic_cast<TT*>(a.get())->get();
            if (binary)
            {
                fwrite(pa, sizeof(NT), na, ofile);
            }
            else
            {
                char fmt_delim[32];
                snprintf(fmt_delim, 32, " %s", teca_vtk_util::vtk_tt<NT>::fmt());
                fprintf(ofile, teca_vtk_util::vtk_tt<NT>::fmt(), pa[0]);
                for (size_t i = 1; i < na; ++i)
                    fprintf(ofile, fmt_delim, pa[i]);
            }
            )
        fprintf(ofile, "\n");
    }
    else
    {
        fprintf(ofile, "0\n");
    }
}

// **************************************************************************
int write_vtk_legacy_header(FILE *ofile,
    const const_p_teca_variant_array &x, const const_p_teca_variant_array &y,
    const const_p_teca_variant_array &z, bool binary,
    const std::string &comment = "")
{
    if (!x && !y && !z)
    {
        TECA_ERROR("data must be at least 1 dimensional")
        return -1;
    }

    fprintf(ofile, "# vtk DataFile Version 2.0\n");

    if (comment.empty())
    {
        time_t rawtime;
        time(&rawtime);
        struct tm *timeinfo = localtime(&rawtime);

        char date_str[128] = {'\0'};
        strftime(date_str, 128, "%F %T", timeinfo);

        fprintf(ofile, "TECA cartesian_mesh_writer "
            TECA_VERSION_DESCR " %s\n", date_str);
    }
    else
    {
        fprintf(ofile, "%s\n", comment.c_str());
    }

    size_t nx = (x ? x->size() : 1);
    size_t ny = (y ? y->size() : 1);
    size_t nz = (z ? z->size() : 1);

    const char *coord_type_str = nullptr;
    TEMPLATE_DISPATCH(const teca_variant_array_impl,
        (x ? x.get() : (y ? y.get() : (z ? z.get() : nullptr))),
        coord_type_str = teca_vtk_util::vtk_tt<NT>::str();
        )

    fprintf(ofile, "%s\n"
        "DATASET RECTILINEAR_GRID\n"
        "DIMENSIONS %zu %zu %zu\n"
        "X_COORDINATES %zu %s\n",
        (binary ? "BINARY" : "ASCII"),
        nx, ny, nz, nx, coord_type_str);

    internals::write_vtk_array_data(ofile, x, binary);

    fprintf(ofile, "\nY_COORDINATES %zu %s\n",
        ny, coord_type_str);

    internals::write_vtk_array_data(ofile, y, binary);

    fprintf(ofile, "\nZ_COORDINATES %zu %s\n",
        nz, coord_type_str);

    internals::write_vtk_array_data(ofile, z, binary);

    return 0;
}

// **************************************************************************
enum center_t { cell, point, face, edge };

int write_vtk_legacy_attribute(FILE *ofile,
    const const_p_teca_array_collection &data, center_t cen, bool binary)
{
    size_t n_arrays = data->size();

    if (!n_arrays)
        return 0;

    const char *att_type_str;
    switch (cen)
    {
    case center_t::cell: att_type_str = "CELL"; break;
    case center_t::point: att_type_str = "POINT"; break;
    default: att_type_str = "FIELD"; break;
    }

    unsigned long n_elem = data->get(0)->size();

    fprintf(ofile, "%s_DATA %zu\nFIELD FieldData %zu\n",
        att_type_str, n_elem, n_arrays);

    for (size_t i = 0; i < n_arrays; ++i)
    {
        const_p_teca_variant_array array = data->get(i);
        std::string array_name = data->get_name(i);

        if (array_name.empty())
            fprintf(ofile, "array_%zu 1 %zu ", i, n_elem);
        else
            fprintf(ofile, "%s 1 %zu ", array_name.c_str(), n_elem);

        TEMPLATE_DISPATCH(const teca_variant_array_impl,
            array.get(), fprintf(ofile, "%s\n",
            teca_vtk_util::vtk_tt<NT>::str());
            )
        else
        {
            TECA_ERROR("unsupported type encountered")
            return -1;
        }

        internals::write_vtk_array_data(ofile, array, binary);

        fprintf(ofile, "\n");
    }

    return 0;
}

// write VTK XML
// ********************************************************************************
int write_vtr(const_p_teca_cartesian_mesh mesh, const std::string &file_name,
    long index, double time, int binary)
{
#if defined(TECA_HAS_VTK) || defined(TECA_HAS_PARAVIEW)
    vtkRectilinearGrid *rg = vtkRectilinearGrid::New();

    if (teca_vtk_util::deep_copy(rg, mesh))
    {
        TECA_ERROR("Failed to copy to vtkRectilinearGrid")
        return -1;
    }

    std::string out_file = file_name;
    teca_file_util::replace_timestep(out_file, index);
    teca_file_util::replace_extension(out_file, "vtr");

    vtkXMLRectilinearGridWriter *w = vtkXMLRectilinearGridWriter::New();
    if (binary)
    {
        w->SetDataModeToAppended();
        w->SetEncodeAppendedData(0);
    }
    w->SetFileName(out_file.c_str());
    w->SetInputData(rg);
    w->Write();

    w->Delete();
    rg->Delete();

    // write a pvd file to capture time coordinate
    std::string pvd_file_name;
    size_t pos;
    if (!(((pos = file_name.find("_%t%")) != std::string::npos) ||
       ((pos = file_name.find("%t%")) != std::string::npos)))
        pos = file_name.rfind(".");

    pvd_file_name = file_name.substr(0, pos);
    pvd_file_name.append(".pvd");

    if (teca_file_util::file_exists(pvd_file_name.c_str()))
    {
        // add dataset to the file
        ofstream pvd_file(pvd_file_name,
            std::ios_base::in|std::ios_base::out|std::ios_base::ate);

        if (!pvd_file.good())
        {
            TECA_ERROR("Failed to open \"" << pvd_file_name << "\"")
            return -1;
        }

        long eof = pvd_file.tellp();
        pvd_file.seekp(eof - 25);

        pvd_file << "<DataSet timestep=\"" << time << "\" group=\"\" part=\"0\" "
            "file=\"" << out_file << "\"/>\n"
            "</Collection>\n"
            "</VTKFile>" << std::endl;

        pvd_file.close();
    }
    else
    {
        // write the initial file
        ofstream pvd_file(pvd_file_name, std::ios_base::out|std::ios_base::trunc);
        if (!pvd_file.good())
        {
            TECA_ERROR("Failed to open \"" << pvd_file_name << "\"")
            return -1;
        }

        pvd_file << "<?xml version=\"1.0\"?>\n"
               "<VTKFile type=\"Collection\" version=\"0.1\"\n"
               "byte_order=\"LittleEndian\" compressor=\"\">\n"
               "<Collection>\n"
               "<DataSet timestep=\"" << time << "\" group=\"\" part=\"0\" "
               "file=\"" << out_file << "\"/>\n"
               "</Collection>\n"
               "</VTKFile>" << std::endl;

        pvd_file.close();
    }

    return 0;
#else
    (void)mesh;
    (void)file_name;
    (void)index;
    (void)time;
    (void)binary;
    return -1;
#endif
}

// write VTK legacy format
// ********************************************************************************
int write_vtk(const_p_teca_cartesian_mesh mesh, const std::string &file_name,
    long index, int binary)
{
    // built without VTK. write as legacy file
    std::string out_file = file_name;
    teca_file_util::replace_timestep(out_file, index);
    teca_file_util::replace_extension(out_file, "vtk");

    const char *mode = binary ? "wb" : "w";

    FILE *ofile = fopen(out_file.c_str(), mode);
    if (!ofile)
    {
        const char *err_desc = strerror(errno);
        TECA_ERROR("Failed to open \"" << out_file << "\""
             << std::endl << err_desc)
        return -1;
    }

    if (internals::write_vtk_legacy_header(ofile,
        mesh->get_x_coordinates(), mesh->get_y_coordinates(),
        mesh->get_z_coordinates(), binary))
    {
        TECA_ERROR("failed to write the header")
        return -1;
    }

    if (internals::write_vtk_legacy_attribute(ofile,
        mesh->get_point_arrays(), internals::center_t::point,
        binary))
    {
        TECA_ERROR("failed to write point arrays")
        return -1;
    }

    if (internals::write_vtk_legacy_attribute(ofile,
        mesh->get_cell_arrays(), internals::center_t::cell,
        binary))
    {
        TECA_ERROR("failed to write point arrays")
        return -1;
    }

    return 0;
}

// ********************************************************************************
int write_bin(const_p_teca_cartesian_mesh mesh, const std::string &file_name,
    long index)
{
    std::string out_file = file_name;
    teca_file_util::replace_timestep(out_file, index);
    teca_file_util::replace_extension(out_file, "bin");

    // serialize the table to a binary representation
    teca_binary_stream bs;
    mesh->to_stream(bs);

    if (teca_file_util::write_stream(out_file.c_str(), "teca_cartesian_mesh", bs))
    {
        TECA_ERROR("Failed to write \"" << out_file << "\"")
        return -1;
    }

    return 0;
}


};

// --------------------------------------------------------------------------
teca_cartesian_mesh_writer::teca_cartesian_mesh_writer()
    : file_name(""), binary(1), output_format(format_auto)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_cartesian_mesh_writer::~teca_cartesian_mesh_writer()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_cartesian_mesh_writer::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_cartesian_mesh_writer":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, file_name,
            "path/name to write series to. supports e,t substitutions")
        TECA_POPTS_GET(int, prefix,binary,
            "if set write VTK binary formats")
        TECA_POPTS_GET(int, prefix, output_format,
            "output file format enum, 0:bin, 1:vtk, 2:vtr, 3:auto."
            "if auto is used, format is deduced from file_name")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh_writer::set_properties(
    const std::string &prefix, variables_map &opts)
{
    TECA_POPTS_SET(opts, std::string, prefix, file_name)
    TECA_POPTS_SET(opts, int, prefix, output_format)
    TECA_POPTS_SET(opts, int, prefix, binary)
}
#endif

// --------------------------------------------------------------------------
const_p_teca_dataset teca_cartesian_mesh_writer::execute(
    unsigned int port, const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
    (void)port;

    const_p_teca_cartesian_mesh mesh
        = std::dynamic_pointer_cast<const teca_cartesian_mesh>(
            input_data[0]);

    // only rank 0 is required to have data
    int rank = 0;
#if defined(TECA_HAS_MPI)
    int init = 0;
    MPI_Initialized(&init);
    if (init)
        MPI_Comm_rank(this->get_communicator(), &rank);
#endif
    if (!mesh)
    {
        if (rank == 0)
        {
            TECA_ERROR("empty input")
        }
        return nullptr;
    }

    const teca_metadata &md = mesh->get_metadata();

    std::string index_request_key;
    if (md.get("index_request_key", index_request_key))
    {
        TECA_ERROR("Dataset metadata is missing the index_request_key key")
        return nullptr;
    }

    unsigned long index = 0;
    if (md.get(index_request_key, index))
    {
        TECA_ERROR("Dataset metadata is missing the \""
            << index_request_key << "\" key")
        return nullptr;
    }

    double time = 0.0;
    if (mesh->get_time(time) &&
        request.get("time", time))
    {
        TECA_ERROR("request missing \"time\"")
        return nullptr;
    }

    // format output file name
    std::string out_file = this->file_name;

    // replace extension
    int fmt = this->output_format;
    if (fmt == format_auto)
    {
        if (out_file.rfind(".vtr") != std::string::npos)
        {
#if !defined(TECA_HAS_VTK) || !defined(TECA_HAS_PARAVIEW)
            TECA_ERROR("writing to vtr format requires VTK or ParaView")
            return nullptr;
#else
            fmt = format_vtr;
#endif
        }
        else if (out_file.rfind(".vtk") != std::string::npos)
        {
            fmt = format_vtk;
        }
        else if (out_file.rfind(".bin") != std::string::npos)
        {
            fmt = format_bin;
        }
        else
        {
            if (rank == 0)
            {
                TECA_WARNING("Failed to determine extension from file name \""
                    << out_file << "\". Using bin format.")
            }
            fmt = format_bin;
        }
    }

    switch (fmt)
    {
        case format_bin:
            internals::write_bin(mesh, this->file_name, index);
            break;
        case format_vtk:
            internals::write_vtk(mesh, this->file_name, index, this->binary);
            break;
        case format_vtr:
            internals::write_vtr(mesh, this->file_name, index, time, this->binary);
            break;
        default:
            TECA_ERROR("Invalid output format")
            return nullptr;
    }

    return mesh;
}
