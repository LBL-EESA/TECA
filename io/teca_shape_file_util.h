#ifndef teca_shape_file_util_h
#define teca_shape_file_util_h

#include "teca_config.h"
#include "teca_binary_stream.h"
#include "teca_geometry.h"
#include "teca_mpi.h"

#include <string>
#include <vector>
#include <ostream>
#include <shapefil.h>

/// @file

namespace teca_shape_file_util
{
/// Get a string with the name of the shape type
TECA_EXPORT
const char *shape_type_name(int shpt);

/// Send the shape object to the stream in a human readable form
TECA_EXPORT
std::ostream &operator<<(std::ostream &os, const SHPObject &obj);

/// Read polygons from the given file.
template<typename coord_t>
TECA_EXPORT
int load_polygons(const std::string &filename,
    std::vector<teca_geometry::polygon<coord_t>> &polys,
    int normalize = 0, int verbose = 0)
{
    if (verbose)
        TECA_STATUS("Opening shape file \"" <<  filename << "\"")

    SHPHandle h = SHPOpen(filename.c_str(), "rb");
    if (h == nullptr)
    {
        TECA_ERROR("Failed to open shape file  \"" << filename << "\"")
        return -1;
    }

    int n_shapes = 0;
    int types = 0;
    double x0[4] = {0.0};
    double x1[4] = {0.0};

    SHPGetInfo(h, &n_shapes, &types, x0, x1);

    if (verbose)
        TECA_STATUS("Found " << n_shapes << " shapes of type "
            << shape_type_name(types))

    if (types != SHPT_POLYGON)
    {
        TECA_ERROR("The shapes contained in \"" << filename
            << "\" are not polygons.")
        SHPClose(h);
        return -1;
    }

    polys.reserve(n_shapes);

    for (int i = 0; i < n_shapes; ++i)
    {
        if (verbose > 1)
           std::cerr << "Reading object " << i << " ... " << std::endl;

        SHPObject *obj = SHPReadObject(h, i);
        if (obj == nullptr)
        {
            TECA_ERROR("Failed to read the " << i << "th polygon")
            SHPClose(h);
            return -1;
        }

        if (verbose > 1)
            std::cerr << "Object" << i << " = " << *obj << std::endl;

        // process each part
        std::vector<int> starts(obj->panPartStart, obj->panPartStart + obj->nParts);
        std::vector<int> ends(obj->panPartStart + 1, obj->panPartStart + obj->nParts);
        ends.push_back(obj->nVertices);
        for (int j = 0; j < obj->nParts; ++j)
        {
            // only process polygons.
            if (obj->panPartType[j] != SHPT_POLYGON)
            {
                TECA_WARNING("Part " << j << " of object " << i << " is "
                    << shape_type_name(obj->panPartType[j])
                    << " but a SHPT_POLYGON is required")
                continue;
            }

            int start = starts[j];
            int n_verts = ends[j] - start;

            // make a polygon
            teca_geometry::polygon<coord_t> poly;
            poly.copy(obj->padfX + start, obj->padfY + start, n_verts);

            if (normalize)
                poly.normalize_coordinates();

            // add it to the collection
            polys.push_back(poly);
        }

        SHPDestroyObject(obj);
    }

    SHPClose(h);

    return 0;
}

/** Read polygons from the given file. MPI rank 0 reads and broadcasts to the
 * other ranks in the job
 */
template<typename coord_t>
TECA_EXPORT
int load_polygons(MPI_Comm comm, const std::string &filename,
    std::vector<teca_geometry::polygon<coord_t>> &polys,
    int normalize = 0, int verbose = 0)
{
    int rank = 0;
#if defined(TECA_HAS_MPI)
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
    {
        MPI_Comm_rank(comm, &rank);
    }
#endif

    teca_binary_stream bs;

    if (rank == 0)
    {
        // read the shape file
        if (teca_shape_file_util::load_polygons(filename, polys, normalize, verbose))
        {
            TECA_ERROR("Failed to read polygons from \""
                << filename << "\"")

            // returning here would introduce an MPI deadlock
        }

        // serialize
        size_t n_polys = polys.size();

        bs.pack(n_polys);

        for (size_t i = 0; i < n_polys; ++i)
            polys[i].to_stream(bs);
    }

    // distribute to all ranks
    bs.broadcast(comm);

    if (rank != 0)
    {
        // deserialize
        size_t n_polys = 0;
        bs.unpack(n_polys);

        polys.resize(n_polys);

        for (size_t i = 0; i < n_polys; ++i)
            polys[i].from_stream(bs);
    }

    return 0;
}

};

#endif
