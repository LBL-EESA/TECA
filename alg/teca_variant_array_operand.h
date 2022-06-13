#ifndef teca_variant_array_operand_h
#define teca_variant_array_operand_h

/// @file

#include "teca_config.h"
#include "teca_variant_array.h"
#include "teca_array_collection.h"

/// Codes dealing with the operands of run time specified operations on teca_variant_array
namespace teca_variant_array_operand
{

/** @brief
 * Handles conversion of literals to variant_arrays and the name resolution of
 * variables.
 */
class TECA_EXPORT resolver
{
public:
    /**  given a text representation of a numeric value in s convert and
     * return a variant_array filled with the numeric value. the type is
     * determined by the last 1 or 2 characters in the string, Valid type
     * codes are:
     *
     *      d -- double
     *      f -- float
     *      L -- long long
     *      l -- long
     *      i -- int
     *      s -- short
     *      c -- char
     *      u -- unsigned, augments any of the integer types
     *
     * the return is non-zero if an error occurred, zero otherwise.
     */
    int get_constant(const char *s, p_teca_variant_array &c);

    /**  given the name of a variable in var_name, set var to point to the
     * array of corresponding name. see set/get_variables.
     */
    int get_variable(const char *var_name,
        const_p_teca_variant_array &var)
    {
        var = m_variables->get(var_name);
        if (!var) return -1;
        return 0;
    }

    /// get the set of arrays used for variable name resolution.
    const_p_teca_array_collection get_variables()
    { return m_variables; }

    /// set the set of arrays used for variable name resolution.
    void set_variables(const_p_teca_array_collection v)
    { m_variables = v; }

private:
    const_p_teca_array_collection m_variables;
};

};

#endif
