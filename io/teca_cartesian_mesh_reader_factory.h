#ifndef teca_cartesian_mesh_reader_factory_h
#define teca_cartesian_mesh_reader_factory_h

/// @file

#include "teca_config.h"
#include "teca_algorithm.h"

/// A factory for Cartesian mesh readers
struct TECA_EXPORT teca_cartesian_mesh_reader_factory
{
    /** creates and initialized a reader from a given file name or regular
     * expression.  the file extension is examined, to determine the type of
     * reader to create. the supported extentsions and the associated readers
     * are as follows:
     *
     *      bin -- teca_cartesian_mesh_reader
     *      nc  -- teca_cf_reader
     *      mcf -- teca_multi_cf_reader
     *
     * returns a new instance of the reader with the file or regex set or a
     * nullptr if an error is encountered
     */
    static p_teca_algorithm New(const std::string &file);
};

#endif
