#include "teca_table_remove_rows.h"

#include "teca_table.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"
#include "teca_parser.h"
#include "teca_variant_array_operator.h"
#include "teca_variant_array_operand.h"

#include <iostream>
#include <string>
#include <set>
#include <sstream>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif
#if defined(TECA_HAS_UDUNITS)
#include "calcalcs.h"
#endif
#if defined(TECA_HAS_MPI)
#include <mpi.h>
#endif

using std::cerr;
using std::endl;

using operator_resolver_t = teca_variant_array_operator::resolver;
using operand_resolver_t = teca_variant_array_operand::resolver;

// --------------------------------------------------------------------------
teca_table_remove_rows::teca_table_remove_rows() :
    remove_dependent_variables(0)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_table_remove_rows::~teca_table_remove_rows()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_table_remove_rows::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_table_remove_rows":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, mask_expression,
            "the expression to create the mask from")
        TECA_POPTS_GET(int, prefix, remove_dependent_variables,
            "when set columns used in the calculation are removed from output")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_table_remove_rows::set_properties(
    const std::string &prefix, variables_map &opts)
{
    TECA_POPTS_SET(opts, std::string, prefix, mask_expression)
    TECA_POPTS_SET(opts, int, prefix, remove_dependent_variables)
}
#endif

// --------------------------------------------------------------------------
void teca_table_remove_rows::set_mask_expression(const std::string &expr)
{
    if (expr == this->mask_expression)
        return;

    // convert the mask_expression to postfix, doing so here let's
    // us know which variables will be needed
    std::set<std::string> dep_vars;
    char *pfix_expr = teca_parser::infix_to_postfix(expr.c_str(), &dep_vars);
    if (!pfix_expr)
    {
        TECA_ERROR("failed to convert \"" << expr << "\" to postfix")
        return;
    }

    this->mask_expression = expr;
    this->postfix_expression = pfix_expr;
    this->dependent_variables = std::move(dep_vars);
    this->set_modified();

    free(pfix_expr);
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_table_remove_rows::execute(
    unsigned int port, const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id() << "teca_table_remove_rows::execute" << endl;
#endif
    (void)port;
    (void)request;

    // get the input table
    const_p_teca_table in_table
        = std::dynamic_pointer_cast<const teca_table>(input_data[0]);

    // only rank 0 is required to have data
    int rank = 0;
#if defined(TECA_HAS_MPI)
    int init = 0;
    MPI_Initialized(&init);
    if (init)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    if (!in_table)
    {
        if (rank == 0)
        {
            TECA_ERROR("Input is empty or not a table")
        }
        return nullptr;
    }

    // grab the aviable vairables from the output
    const_p_teca_array_collection variables = in_table->get_columns();

    // construct the data adaptor that serves up variables
    // and constants
    operand_resolver_t operand_resolver;
    operand_resolver.set_variables(variables);

    // verify that we have a valid expression
    if (this->postfix_expression.empty())
    {
        TECA_ERROR("An expression was not provided.")
        return nullptr;
    }

    // evaluate the postfix expression
    const_p_teca_variant_array mask;
    if (teca_parser::eval_postfix<p_teca_variant_array,
        const_p_teca_variant_array, operand_resolver_t,
        operator_resolver_t>(mask, this->postfix_expression.c_str(),
            operand_resolver))
    {
        TECA_ERROR("failed to evaluate the mask expression \""
            << this->mask_expression << "\"")
        return nullptr;
    }

    // construct the output
    p_teca_table out_table = teca_table::New();
    out_table->copy_metadata(in_table);
    out_table->copy_structure(in_table);

    unsigned long n_rows = in_table->get_number_of_rows();
    unsigned long n_cols = in_table->get_number_of_columns();

    // identify rows that meet all criteria
    std::vector<unsigned long> valid_rows;
    valid_rows.reserve(n_rows);

    TEMPLATE_DISPATCH(const teca_variant_array_impl,
        mask.get(),
        const NT *pmask = static_cast<TT*>(mask.get())->get();
        for (unsigned long i = 0; i < n_rows; ++i)
        {
            if (!pmask[i])
                valid_rows.push_back(i);
        }
        )

    // for each column copy the valid rows
    unsigned long n_valid = valid_rows.size();
    for (unsigned long j = 0; j < n_cols; ++j)
    {
        const_p_teca_variant_array in_col = in_table->get_column(j);

        p_teca_variant_array out_col = out_table->get_column(j);
        out_col->resize(n_valid);

        TEMPLATE_DISPATCH(teca_variant_array_impl,
            out_col.get(),
            const NT *pin = static_cast<const TT*>(in_col.get())->get();
            NT *pout = static_cast<TT*>(out_col.get())->get();
            for (unsigned long i = 0; i < n_valid; ++i)
            {
                pout[i] = pin[valid_rows[i]];
            }
            )
    }

    // remove dependent variables
    if (this->remove_dependent_variables)
    {
        std::set<std::string>::iterator it = this->dependent_variables.begin();
        std::set<std::string>::iterator end = this->dependent_variables.end();
        for (; it != end; ++it)
            out_table->remove_column(*it);
    }

    return out_table;
}
