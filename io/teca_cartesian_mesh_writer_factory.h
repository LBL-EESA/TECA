#ifndef teca_cartesian_mesh_writer_factory_h
#define teca_cartesian_mesh_writer_factory_h

#include "teca_config.h"
#include "teca_algorithm.h"

/// A factory for Cartesian mesh writers
struct TECA_EXPORT teca_cartesian_mesh_writer_factory
{
     /** creates and initialized a writer from a given file name
      * or regular expression.  the file extension is examined,
      * to determine the type of writer to create. the supported
      * extentsions and the associated writers are as follows:
      *
      * bin -- teca_cartesian_mesh_writer
      * nc  -- teca_cf_writer
      *
      * returns a new instance of the writer with the file or
      * regex set or a nullptr if an error is encountered
      */
    static p_teca_algorithm New(const std::string &file);
};

#endif
