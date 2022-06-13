#ifndef teca_table_remove_rows_h
#define teca_table_remove_rows_h

#include "teca_config.h"
#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_table_remove_rows)

/** @brief
 * An algorithm that removes rows from a table where
 * a given expression evaluates to true.
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
class TECA_EXPORT teca_table_remove_rows : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_table_remove_rows)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_table_remove_rows)
    TECA_ALGORITHM_CLASS_NAME(teca_table_remove_rows)
    ~teca_table_remove_rows();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @name mask_expression
     * set the expression to use to determine which rows are removed. rows are
     * removed where the expression evaluates true.
     */
    ///@{
    /// Set the mask expression
    void set_mask_expression(const std::string &expr);

    /// Get the mask expression
    std::string get_mask_expression()
    { return this->mask_expression; }
    ///@}

    /** @name remove_dependent_variables
     * when set columns used in the calculation are removed from the output.
     * default off.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, remove_dependent_variables)
    ///@}


protected:
    teca_table_remove_rows();

private:
    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    std::string mask_expression;
    std::string postfix_expression;
    std::set<std::string> dependent_variables;
    int remove_dependent_variables;
};

#endif
