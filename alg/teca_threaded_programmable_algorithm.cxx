#include "teca_threaded_programmable_algorithm.h"

#include "teca_dataset.h"

using std::cerr;
using std::endl;

// --------------------------------------------------------------------------
teca_threaded_programmable_algorithm::teca_threaded_programmable_algorithm()
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);

    // install default callbacks
    this->use_default_report_action();
    this->use_default_request_action();
    this->use_default_execute_action();

    strncpy(this->class_name, "teca_threaded_programmable_algorithm",
        sizeof(this->class_name));
}

// --------------------------------------------------------------------------
teca_threaded_programmable_algorithm::~teca_threaded_programmable_algorithm()
{}

// --------------------------------------------------------------------------
int teca_threaded_programmable_algorithm::set_name(const std::string &name)
{
    if (snprintf(this->class_name, sizeof(this->class_name),
        "teca_threaded_programmable_algorithm(%s)", name.c_str()) >=
        static_cast<int>(sizeof(this->class_name)))
    {
        TECA_ERROR("name is too long for the current buffer size "
            << sizeof(this->class_name))
        return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
void teca_threaded_programmable_algorithm::use_default_report_action()
{
    this->set_report_callback(
        [](unsigned int, const std::vector<teca_metadata> &input_md)
            -> teca_metadata
        {
            // the default implementation passes meta data through
            if (input_md.size())
                return input_md[0];
            return teca_metadata();
        });
}

// --------------------------------------------------------------------------
void teca_threaded_programmable_algorithm::use_default_request_action()
{
    this->set_request_callback(
        [this](unsigned int, const std::vector<teca_metadata> &,
            const teca_metadata &request) -> std::vector<teca_metadata>
        {
            // default implementation forwards request upstream
            return std::vector<teca_metadata>(
                this->get_number_of_input_connections(), request);
        });
}

// --------------------------------------------------------------------------
void teca_threaded_programmable_algorithm::use_default_execute_action()
{
    this->set_execute_callback(
        [] (unsigned int, const std::vector<const_p_teca_dataset> &input_data,
            const teca_metadata &, int) -> const_p_teca_dataset
        {
            // default implementation makes a shallow copy of the first
            // input dataset
            if (input_data.size() < 1)
                return p_teca_dataset();

            p_teca_dataset output_data = input_data[0]->new_instance();

            output_data->shallow_copy(
                std::const_pointer_cast<teca_dataset>(input_data[0]));

            return output_data;
        });
}

// --------------------------------------------------------------------------
teca_metadata teca_threaded_programmable_algorithm::get_output_metadata(
    unsigned int port, const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_threaded_programmable_algorithm::get_output_metadata" << endl;
#endif

    return this->report_callback(port, input_md);
}

// --------------------------------------------------------------------------
std::vector<teca_metadata>
teca_threaded_programmable_algorithm::get_upstream_request(
    unsigned int port, const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_threaded_programmable_algorithm::get_upstream_request" << endl;
#endif

    return this->request_callback(port, input_md, request);
}


// --------------------------------------------------------------------------
const_p_teca_dataset teca_threaded_programmable_algorithm::execute(
    unsigned int port, const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request, int streaming)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_threaded_programmable_algorithm::execute" << endl;
#endif

    return this->execute_callback(port, input_data, request, streaming);
}
