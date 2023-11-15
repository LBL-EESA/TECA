#include "teca_cartesian_mesh_writer.h"

#include "teca_config.h"
#include "teca_cartesian_mesh.h"
#include "teca_curvilinear_mesh.h"
#include "teca_arakawa_c_grid.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_file_util.h"
#include "teca_vtk_util.h"
#include "teca_metadata_util.h"

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

using namespace teca_variant_array_util;

namespace internals {

bool is_big_endian(void)
{
    union { uint32_t i; char c[4]; } bint = {0x01020304};
    return bint.c[0] == 1;
}

template<typename num_t>
int fwrite_big_endian(num_t *data, size_t elem_size, size_t n_elem, FILE *ofile)
{
    size_t elem_size_2 = elem_size / 2;
    size_t n_bytes = elem_size*n_elem;
    char *tmp = (char*)malloc(n_bytes);
    char *dst = tmp;
    char *src = (char*)data;
    for (size_t i = 0; i < n_elem; ++i)
    {
        for (size_t j = 0; j < elem_size_2; ++j)
        {
            size_t jj = elem_size - j - 1;
            dst[j] = src[jj];
            dst[jj] = src[j];
        }
        dst += elem_size;
        src += elem_size;
    }
    if (fwrite(tmp, elem_size, n_elem, ofile) != n_elem)
    {
        free(tmp);
        char *estr = strerror(errno);
        TECA_ERROR("Failed to fwrite data. "  << estr)
        return -1;

    }
    free(tmp);
    return 0;
}

template<typename num_t>
int fwrite_native_endian(num_t *data, size_t elem_size, size_t n_elem, FILE *ofile)
{
    if (fwrite(data, elem_size, n_elem, ofile) != n_elem)
    {
        char *estr = strerror(errno);
        TECA_ERROR("Failed to fwrite data. "  << estr)
        return -1;
    }
    return 0;
}

void write_vtk_array_data(FILE *ofile,
    const const_p_teca_variant_array &a, int binary)
{
    if (a)
    {
        size_t na = a->size();
        VARIANT_ARRAY_DISPATCH(a.get(),
            auto [spa, pa] = get_host_accessible<CTT>(a);
            sync_host_access_any(a);
            if (binary)
            {
                // because VTK's legacy file fomrat  requires big endian storage
                if ((sizeof(NT) > 1) && !is_big_endian())
                    fwrite_big_endian(pa, sizeof(NT), na, ofile);
                else
                    fwrite_native_endian(pa, sizeof(NT), na, ofile);
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
        fprintf(ofile, "METADATA\n");
        fprintf(ofile, "INFORMATION 0\n\n");
    }
    else
    {
        TECA_ERROR("Attempt to write a nullptr")
        fprintf(ofile, "0\n");
    }
}

// **************************************************************************
int write_vtk_legacy_rectilinear_header(FILE *ofile,
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
    VARIANT_ARRAY_DISPATCH(
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
int write_vtk_legacy_arakawa_c_grid_header(FILE *ofile,
    size_t *extent, const const_p_teca_variant_array &x,
    const const_p_teca_variant_array &y, const const_p_teca_variant_array &z,
    bool binary, const std::string &comment = "")
{
    // this is because the reader should always provide 3D data
    if (!x || !y || !z)
    {
        TECA_ERROR("data must be 3 dimensional")
        return -1;
    }

    size_t nx = extent[1] - extent[0] + 1;
    size_t ny = extent[3] - extent[2] + 1;
    size_t nz = extent[5] - extent[4] + 1;
    size_t nxy = nx*ny;
    size_t nxyz = nxy*nz;

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

    const char *coord_type_str = nullptr;
    VARIANT_ARRAY_DISPATCH(
        (x ? x.get() : (y ? y.get() : (z ? z.get() : nullptr))),
        coord_type_str = teca_vtk_util::vtk_tt<NT>::str();
        )

    fprintf(ofile, "%s\n"
        "DATASET STRUCTURED_GRID\n"
        "DIMENSIONS %zu %zu %zu\n"
        "POINTS %zu %s\n",
        (binary ? "BINARY" : "ASCII"),
        nx, ny, nz, nxyz, coord_type_str);

    // convert the WRF coordinate arrays into VTK layout
    p_teca_variant_array xyz = x->new_instance(3*nxyz);
    VARIANT_ARRAY_DISPATCH(x.get(),

        assert_type<CTT>(y, z);

        auto [pxyz] = data<TT>(xyz);
        auto [spx, px, spy, py, spz, pz] = get_host_accessible<CTT>(x, y, z);

        sync_host_access_any(x, y, z);

        for (size_t k = 0; k < nz; ++k)
        {
            NT z = pz[k];
            size_t kk = k*nxy;
            for (size_t j = 0; j < ny; ++j)
            {
                size_t jj = j*nx;
                size_t kk_jj = kk + jj;
                for (size_t i = 0; i < nx; ++i)
                {
                    size_t q = 3*(kk_jj + i);
                    pxyz[q    ] = px[jj + i];
                    pxyz[q + 1] = py[jj + i];
                    pxyz[q + 2] = z;
                }
            }
        }
        )

    internals::write_vtk_array_data(ofile, xyz, binary);

    return 0;
}

// **************************************************************************
int write_vtk_legacy_curvilinear_header(FILE *ofile,
    size_t *extent, const const_p_teca_variant_array &x,
    const const_p_teca_variant_array &y, const const_p_teca_variant_array &z,
    bool binary, const std::string &comment = "")
{
    // this is because the reader should always provide 3D data
    if (!x || !y)
    {
        TECA_ERROR("data must be 2 dimensional")
        return -1;
    }

    size_t nx = extent[1] - extent[0] + 1;
    size_t ny = extent[3] - extent[2] + 1;
    size_t nz = extent[5] - extent[4] + 1;
    size_t nxy = nx*ny;
    size_t nxyz = nxy*nz;

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

    const char *coord_type_str = nullptr;
    VARIANT_ARRAY_DISPATCH(
        (x ? x.get() : (y ? y.get() : (z ? z.get() : nullptr))),
        coord_type_str = teca_vtk_util::vtk_tt<NT>::str();
        )

    fprintf(ofile, "%s\n"
        "DATASET STRUCTURED_GRID\n"
        "DIMENSIONS %zu %zu %zu\n"
        "POINTS %zu %s\n",
        (binary ? "BINARY" : "ASCII"),
        nx, ny, nz, nxyz, coord_type_str);

    // convert the WRF arrays into VTK layout
    p_teca_variant_array xyz = x->new_instance(3*nxyz);

    VARIANT_ARRAY_DISPATCH(x.get(),

        assert_type<CTT>(y, z);

        auto [pxyz] = data<TT>(xyz);
        auto [spx, px, spy, py] = get_host_accessible<CTT>(x, y);

        CSP spz;
        const NT *pz = nullptr;
        if (z)
            std::tie(spz, pz) = get_host_accessible<CTT>(z);

        sync_host_access_any(x, y);

        for (size_t i = 0; i < nxyz; ++i)
            pxyz[3*i] = px[i];

        for (size_t i = 0; i < nxyz; ++i)
            pxyz[3*i + 1] = py[i];

        for (size_t i = 0; i < nxyz; ++i)
            pxyz[3*i + 2] = pz ? pz[i] : NT(0);
        )

    internals::write_vtk_array_data(ofile, xyz, binary);

    return 0;
}

// **************************************************************************
enum center_t { cell, point, face, edge };

int write_vtk_legacy_attribute(FILE *ofile, unsigned long n_vals,
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

    fprintf(ofile, "%s_DATA %zu\nFIELD FieldData %zu\n",
        att_type_str, n_vals, n_arrays);

    for (size_t i = 0; i < n_arrays; ++i)
    {

        unsigned long n_elem = data->get(i)->size();
        if (n_elem != n_vals)
            continue;

        const_p_teca_variant_array array = data->get(i);
        std::string array_name = data->get_name(i);

        if (array_name.empty())
            fprintf(ofile, "array_%zu 1 %zu ", i, n_elem);
        else
            fprintf(ofile, "%s 1 %zu ", array_name.c_str(), n_elem);

        VARIANT_ARRAY_DISPATCH(array.get(),
            fprintf(ofile, "%s\n", teca_vtk_util::vtk_tt<NT>::str());
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
int write_vtr(const_p_teca_mesh dataset, const std::string &file_name,
    long index, double time, int binary)
{
#if defined(TECA_HAS_VTK) || defined(TECA_HAS_PARAVIEW)
    const_p_teca_cartesian_mesh mesh = std::dyanmic_pointer_cast<const teca_cartesian_mesh>(dataset);
    if (!mesh)
    {
        TECA_ERROR("The vtr format is only supported for type "
            "teca_cartesian_mesh but this is a " << dataset->get_class_name())
        return -1;
    }

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
    (void)dataset;
    (void)file_name;
    (void)index;
    (void)time;
    (void)binary;
    return -1;
#endif
}

// write VTK legacy format
// ********************************************************************************
int write_vtk(const const_p_teca_mesh &mesh, const std::string &file_name,
    long index, int binary)
{
    // built without VTK. write as legacy file
    std::string out_file = file_name;
    teca_file_util::replace_timestep(out_file, index);
    teca_file_util::replace_extension(out_file, "vtk");

    const char *mode = binary ? "wb" : "w";

    int dual_grid = 0;
    unsigned long nx = 0;
    unsigned long ny = 0;
    unsigned long nz = 0;

    FILE *ofile = fopen(out_file.c_str(), mode);
    if (!ofile)
    {
        const char *err_desc = strerror(errno);
        TECA_ERROR("Failed to open \"" << out_file << "\""
             << std::endl << err_desc)
        return -1;
    }

    if (dynamic_cast<const teca_cartesian_mesh *>(mesh.get()))
    {
        const_p_teca_cartesian_mesh r_mesh
             = std::static_pointer_cast<const teca_cartesian_mesh>(mesh);

        size_t extent[6] = {0};
        r_mesh->get_extent(extent);

        if (internals::write_vtk_legacy_rectilinear_header(ofile,
            r_mesh->get_x_coordinates(), r_mesh->get_y_coordinates(),
            r_mesh->get_z_coordinates(), binary))
        {
            TECA_ERROR("failed to write the header")
            return -1;
        }

        nx = extent[1] - extent[0] + 1;
        ny = extent[3] - extent[2] + 1;
        nz = extent[5] - extent[4] + 1;
    }
    else if (std::dynamic_pointer_cast<const teca_curvilinear_mesh>(mesh))
    {
        const_p_teca_curvilinear_mesh c_mesh
             = std::static_pointer_cast<const teca_curvilinear_mesh>(mesh);

        size_t extent[6] = {0};
        c_mesh->get_extent(extent);

        if (internals::write_vtk_legacy_curvilinear_header(ofile,
            extent, c_mesh->get_x_coordinates(), c_mesh->get_y_coordinates(),
            c_mesh->get_z_coordinates(), binary))
        {
            TECA_ERROR("failed to write the header")
            return -1;
        }

        nx = extent[1] - extent[0] + 1;
        ny = extent[3] - extent[2] + 1;
        nz = extent[5] - extent[4] + 1;
    }
    else if (std::dynamic_pointer_cast<const teca_arakawa_c_grid>(mesh))
    {
        const_p_teca_arakawa_c_grid acg_mesh
             = std::static_pointer_cast<const teca_arakawa_c_grid>(mesh);

        size_t extent[6] = {0};
        acg_mesh->get_extent(extent);

        // m coordinates are cell centers and map to VTK point centering
        // on a dual mesh.
        dual_grid = 1;

        if (internals::write_vtk_legacy_arakawa_c_grid_header(ofile,
            extent, acg_mesh->get_m_x_coordinates(), acg_mesh->get_m_y_coordinates(),
            acg_mesh->get_m_z_coordinates(), binary))
        {
            TECA_ERROR("failed to write the header")
            return -1;
        }

        nx = extent[1] - extent[0] + 1;
        ny = extent[3] - extent[2] + 1;
        nz = extent[5] - extent[4] + 1;
    }
    else
    {
        TECA_ERROR("Unsupported mesh type \"" << mesh->get_class_name() << "\"")
        return -1;
    }

    if (dual_grid)
    {
        unsigned long n_vals = nx*ny*nz;

        if (internals::write_vtk_legacy_attribute(ofile, n_vals,
            mesh->get_cell_arrays(), internals::center_t::point,
            binary))
        {
            TECA_ERROR("failed to write point arrays")
            return -1;
        }
    }
    else
    {
        unsigned long n_vals_p = nx*ny*nz;

        if (internals::write_vtk_legacy_attribute(ofile, n_vals_p,
            mesh->get_point_arrays(), internals::center_t::point,
            binary))
        {
            TECA_ERROR("failed to write point arrays")
            return -1;
        }

        unsigned long n_vals_c = (nx > 1 ? nx - 1 : 1)*
            (ny > 1 ? ny - 1 : 1)*(nz > 1 ? nz - 1 : 1);

        if (internals::write_vtk_legacy_attribute(ofile, n_vals_c,
            mesh->get_cell_arrays(), internals::center_t::cell,
            binary))
        {
            TECA_ERROR("failed to write cell arrays")
            return -1;
        }
    }

    fclose(ofile);

    return 0;
}

// ********************************************************************************
int write_bin(const_p_teca_mesh mesh, const std::string &file_name,
    long index)
{
    std::string out_file = file_name;
    teca_file_util::replace_timestep(out_file, index);
    teca_file_util::replace_extension(out_file, "bin");

    // serialize the mesh to a binary representation
    teca_binary_stream bs;
    bs.pack(mesh->get_type_code());
    if (mesh->to_stream(bs))
    {
        TECA_ERROR("Failed to serialize \"" << mesh->get_class_name() << "\"")
        return -1;
    }

    if (teca_file_util::write_stream(out_file.c_str(),
        S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH, "teca_cartesian_mesh_writer_v2", bs))
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

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_cartesian_mesh_writer::set_properties(
    const std::string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

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

    const_p_teca_mesh mesh
        = std::dynamic_pointer_cast<const teca_mesh>(input_data[0]);

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
            TECA_FATAL_ERROR("empty input")
        }
        return nullptr;
    }

    unsigned long index = 0;
    if (mesh->get_request_index(index))
    {
        TECA_FATAL_ERROR("Failed to determine the requested index")
        return nullptr;
    }

    double time = 0.0;
    if (mesh->get_time(time) && request.get("time", time))
    {
        TECA_FATAL_ERROR("Failed to determine the time of index " << index)
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
            TECA_FATAL_ERROR("writing to vtr format requires VTK or ParaView")
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
            TECA_FATAL_ERROR("Invalid output format")
            return nullptr;
    }

    return mesh;
}
