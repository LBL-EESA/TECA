#ifndef teca_evaluate_expression_h
#define teca_evaluate_expression_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"
#include "teca_array_attributes.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_evaluate_expression)

/** @brief
 * An algorithm that evaluates an expression stores the
 * result in a new variable.
 *
 * @details
 * The expression parser supports the following operations:
 *     +,-,*,/,%,<.<=,>,>=,==,!=,&&,||.!,?
 *
 * Grouping in the expression is denoted in the usual
 * way: ()
 *
 * Constants in the expression are expanded to full length
 * arrays and can be typed. The supported types are:
 *     d,f,L,l,i,s,c
 * Corresponding to double,float, long long, long, int,
 * short and char respectively. Integer types can be
 * unsigned by including u after the code.
 */
class TECA_EXPORT teca_evaluate_expression : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_evaluate_expression)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_evaluate_expression)
    TECA_ALGORITHM_CLASS_NAME(teca_evaluate_expression)
    ~teca_evaluate_expression();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @name expression
     * Set the expression to evaluate.
     */
    ///@{
    /// Set the expression.
    void set_expression(const std::string &expr);

    /// Get the expression.
    std::string get_expression()
    { return this->expression; }
    ///@}

    /** @name result_variable
     * set the name of the variable to store the result in
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, result_variable)
    ///@}

    /** @name result_attributes
     * set the NetCDF attributes of the result. This is needed for
     * teca_cf_writer and is otherwise optional.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(teca_array_attributes, result_attributes)
    ///@}

    /** @name remove_dependent_variables
     * when set columns used in the calculation are removed from the output.
     * default off.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, remove_dependent_variables)
    ///@}

protected:
    teca_evaluate_expression();

private:
    using teca_algorithm::get_output_metadata;

    teca_metadata get_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    std::vector<teca_metadata> get_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    std::string expression;
    std::string result_variable;
    teca_array_attributes result_attributes;

    std::string postfix_expression;
    std::set<std::string> dependent_variables;
    int remove_dependent_variables;
};

#endif
