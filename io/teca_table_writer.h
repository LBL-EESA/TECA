#ifndef teca_table_writer_h
#define teca_table_writer_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"
#include "teca_table_fwd.h"

#include <vector>
#include <string>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_table_writer)

/**
an algorithm that writes cartesian meshes in VTK format.
*/
class teca_table_writer : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_table_writer)
    ~teca_table_writer();

    // set base filename. if the time step is found
    // then it is appended. the extension will be appeneded,
    // bin when binary mode is enabled, csv otherwise.
    TECA_ALGORITHM_PROPERTY(std::string, base_file_name)

    // enable binary mode. default off. when not in
    // binary mode a csv format is used.
    TECA_ALGORITHM_PROPERTY(bool, binary_mode)

protected:
    teca_table_writer();

    int write_csv(const_p_teca_table table, const std::string &file_name);
    int write_bin(const_p_teca_table table, const std::string &file_name);

private:

    const_p_teca_dataset execute(
        unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    std::string base_file_name;
    bool binary_mode;
};

#endif
