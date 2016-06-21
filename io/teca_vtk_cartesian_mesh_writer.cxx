#include "teca_vtk_cartesian_mesh_writer.h"

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
#if defined(TECA_HAS_PARAVIEW)
#include "vtkSmartPointer.h"
#include "vtkPVXMLElement.h"
#include "vtkPVXMLParser.h"
#endif

#include <iostream>
#include <sstream>
#include <cstring>
#include <cerrno>
#include <string>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

#if !defined(TECA_HAS_VTK) && !defined(TECA_HAS_PARAVIEW)
namespace {

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
    if (!x && !y &&!z)
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

        fprintf(ofile, "TECA vtk_cartesian_mesh_writer "
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

    write_vtk_array_data(ofile, x, binary);

    fprintf(ofile, "\nY_COORDINATES %zu %s\n",
        ny, coord_type_str);

    write_vtk_array_data(ofile, y, binary);

    fprintf(ofile, "\nZ_COORDINATES %zu %s\n",
        nz, coord_type_str);

    write_vtk_array_data(ofile, z, binary);

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

    fprintf(ofile, "%s_DATA %zu\n", att_type_str,
        data->get(0)->size());

    for (size_t i = 0; i < n_arrays; ++i)
    {
        const_p_teca_variant_array array = data->get(i);
        std::string array_name = data->get_name(i);

        fprintf(ofile, "SCALARS");
        if (array_name.empty())
            fprintf(ofile, " array_%zu ", i);
        else
            fprintf(ofile, " %s ", array_name.c_str());

        TEMPLATE_DISPATCH(const teca_variant_array_impl,
            array.get(), fprintf(ofile, "%s",
            teca_vtk_util::vtk_tt<NT>::str());
            )
        else
        {
            TECA_ERROR("unsupported type encountered")
            return -1;
        }

        fprintf(ofile, " 1\n"
            "LOOKUP_TABLE default\n");

        write_vtk_array_data(ofile, array, binary);

        fprintf(ofile, "\n");
    }

    return 0;
}
};
#endif

// --------------------------------------------------------------------------
teca_vtk_cartesian_mesh_writer::teca_vtk_cartesian_mesh_writer()
    : file_name(""), binary(0)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_vtk_cartesian_mesh_writer::~teca_vtk_cartesian_mesh_writer()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_vtk_cartesian_mesh_writer::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_vtk_cartesian_mesh_writer":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix,file_name,
            "path/name to write series to")
        TECA_POPTS_GET(int, prefix,binary,
            "if set write raw binary (ie smaller, faster)")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_vtk_cartesian_mesh_writer::set_properties(
    const std::string &prefix, variables_map &opts)
{
    TECA_POPTS_SET(opts, std::string, prefix, file_name)
    TECA_POPTS_SET(opts, int, prefix, binary)
}
#endif

// --------------------------------------------------------------------------
const_p_teca_dataset teca_vtk_cartesian_mesh_writer::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
    (void)port;

    const_p_teca_cartesian_mesh mesh
        = std::dynamic_pointer_cast<const teca_cartesian_mesh>(
            input_data[0]);

    if (!mesh)
    {
        TECA_ERROR("empty input")
        return nullptr;
    }

    unsigned long time_step = 0;
    if (mesh->get_time_step(time_step) &&
        request.get("time_step", time_step))
    {
        TECA_ERROR("request missing \"time_step\"")
        return nullptr;
    }

    double time = 0.0;
    if (mesh->get_time(time) &&
        request.get("time", time))
    {
        TECA_ERROR("request missing \"time\"")
        return nullptr;
    }

    // if we have VTK then use their XML file formats
    // otherwise fallback to our legacy writer
#if defined(TECA_HAS_VTK) || defined(TECA_HAS_PARAVIEW)
    vtkRectilinearGrid *rg = vtkRectilinearGrid::New();

    if (teca_vtk_util::deep_copy(rg, mesh))
    {
        TECA_ERROR("Failed to copy to vtkRectilinearGrid")
        return nullptr;
    }

    std::string out_file = this->file_name;
    teca_file_util::replace_timestep(out_file, time_step);

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

#if defined(TECA_HAS_PARAVIEW)
    // write a pvd file to capture time coordinate
    std::string pvd_file_name;
    size_t pos;
    if ((pos = this->file_name.find("%t%")) == std::string::npos)
        pos = this->file_name.rfind(".");
    pvd_file_name = this->file_name.substr(0, pos);
    pvd_file_name.append(".pvd");

    vtkPVXMLParser *parser = vtkPVXMLParser::New();
    if (teca_file_util::file_exists(pvd_file_name.c_str()))
    {
        // read the file
        parser->SetFileName(pvd_file_name.c_str());
        if (!parser->Parse())
        {
            TECA_ERROR("failed to parse \"" << pvd_file_name << "\"")
            return nullptr;
        }

        // insert a node for this time step to the tree
        vtkSmartPointer<vtkPVXMLElement> elem
            = vtkSmartPointer<vtkPVXMLElement>::New();

        elem->SetName("DataSet");
        elem->AddAttribute("timestep", time);
        elem->AddAttribute("group", 0);
        elem->AddAttribute("part", 0);
        elem->AddAttribute("file", out_file.c_str());

        vtkPVXMLElement *root = parser->GetRootElement();
        vtkPVXMLElement *coll = root->FindNestedElementByName("Collection");
        coll->AddNestedElement(elem);
        elem->Delete();

        // write the file back out
        ofstream pvd_file(pvd_file_name, std::ios_base::out|std::ios_base::trunc);
        if (!pvd_file.good())
        {
            TECA_ERROR("Failed to open \"" << pvd_file_name << "\"")
            return nullptr;
        }
        root->PrintXML(pvd_file, vtkIndent());
        pvd_file.close();
    }
    else
    {
        // write an initial file
        ofstream pvd_file(pvd_file_name, std::ios_base::out|std::ios_base::trunc);
        if (!pvd_file.good())
        {
            TECA_ERROR("Failed to open \"" << pvd_file_name << "\"")
            return nullptr;
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
#endif
#else
    // built without VTK. write as legacy file
    std::string out_file = this->file_name;
    teca_file_util::replace_timestep(out_file, time_step);

    const char *mode = this->binary ? "wb" : "w";

    FILE *ofile = fopen(out_file.c_str(), mode);
    if (!ofile)
    {
        const char *err_desc = strerror(errno);
        TECA_ERROR("Failed to open \"" << out_file << "\""
             << std::endl << err_desc)
        return nullptr;
    }

    if (write_vtk_legacy_header(ofile, mesh->get_x_coordinates(),
        mesh->get_y_coordinates(), mesh->get_z_coordinates(),
        this->binary))
    {
        TECA_ERROR("failed to write the header")
        return nullptr;
    }

    if (write_vtk_legacy_attribute(ofile, mesh->get_point_arrays(),
        center_t::point, this->binary))
    {
        TECA_ERROR("failed to write point arrays")
        return nullptr;
    }

    if (write_vtk_legacy_attribute(ofile, mesh->get_cell_arrays(),
        center_t::cell, this->binary))
    {
        TECA_ERROR("failed to write point arrays")
        return nullptr;
    }
#endif

    return p_teca_dataset();
}
