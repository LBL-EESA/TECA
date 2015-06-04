#include "teca_table.h"
#include "teca_metadata.h"
#include "teca_programmable_source.h"
#include "teca_table_writer.h"
#include "teca_time_step_executive.h"

#include <iostream>
using namespace std;

struct report
{
    teca_metadata operator()()
    {
        teca_metadata md;
        md.insert("number_of_time_steps", long(4));
        return md;
    }
};

struct execute
{
    const_p_teca_dataset operator()(const teca_metadata &req)
    {
        long step;
        if (req.get("time_step", step))
        {
            cerr << "request is missing \"time_step\"" << endl;
            return nullptr;
        }

        p_teca_table t = teca_table::New();

        t->declare_columns(
            "step", long(), "name", std::string(),
            "age", int(), "skill", double());

        t->reserve(64);

        t->append(step, std::string("James"), 0, 0.0);
        t << step << std::string("Jessie") << 2 << 0.2
          << step << std::string("Frank") << 3 << 3.1415;
        t->append(step, std::string("Mike"), 1, 0.1);
        t << step << std::string("Tom") << 4 << 0.4;

        return t;
    }
};



int main(int, char **)
{
    p_teca_programmable_source s = teca_programmable_source::New();
    s->set_report_function(report());
    s->set_execute_function(execute());

    p_teca_table_writer w = teca_table_writer::New();
    w->set_input_connection(s->get_output_port());
    w->set_executive(teca_time_step_executive::New());
    w->set_file_name("table_writer_test_%t%.%e%");

    w->update();

    return 0;
}
