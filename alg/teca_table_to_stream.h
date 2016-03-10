#ifndef teca_table_to_stream_h
#define teca_table_to_stream_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>
#include <iostream>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_table_to_stream)

/// an algorithm that serializes a table to a c++ stream object.
/// This is primarilly useful for debugging.
class teca_table_to_stream : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_table_to_stream)
    ~teca_table_to_stream();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    TECA_ALGORITHM_PROPERTY(std::string, header)
    TECA_ALGORITHM_PROPERTY(std::string, footer)

    // set the stream object to store the table in.
    // note that this stream must out live it's use here
    // as streams are not copy-able and thus we store
    // a reference to it.
    void set_stream(std::ostream &s);

    // set the stream by name. stderr, stdout.
    void set_stream(const std::string &s);
    void set_stream_to_stderr();
    void set_stream_to_stdout();

protected:
    teca_table_to_stream();

private:
    const_p_teca_dataset execute(
        unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    std::string header;
    std::string footer;
    std::ostream *stream;
};

#endif
