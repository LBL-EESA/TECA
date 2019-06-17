#ifndef teca_parser_h
#define teca_parser_h

#include "teca_common.h"

#include <vector>
#include <set>
#include <string>
#include <iostream>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <cmath>
#include <cstdio>

#define TECA_PARSER_ERROR(_descr, _expr, _pos)                          \
    TECA_MESSAGE(std::cerr, "ERROR:", ANSI_RED,                         \
        << "" _descr                                                    \
        << "at position " << (_pos) << " in  \""                        \
        << std::string(_expr, _expr+(_pos)) << END_HL                   \
        << BEGIN_HL(ANSI_RED) << (_expr+(_pos))[0] << END_HL            \
        << BEGIN_HL(ANSI_WHITE) << std::string(_expr+(_pos)+1) << "\"")

#define TECA_SYNTAX_ERROR(_expr, _pos)                                  \
    TECA_PARSER_ERROR("Syntax error. ", _expr, _pos)

#define TECA_NAME_RESOLUTION_ERROR(_name, _expr, _pos)                  \
    TECA_PARSER_ERROR("Name resolution error \""                        \
        << _name << "\". ", _expr, _pos)

#define TECA_INVALID_OPERATION_ERROR(_op, _expr, _pos)                  \
    TECA_PARSER_ERROR("Invalid operation \""                            \
        << _op << "\". ", _expr, _pos)

#define TECA_OPERATION_FAILED_ERROR(_op, _expr, _pos)                   \
    TECA_PARSER_ERROR("Operation \"" << _op                             \
        << "\" failed. " , _expr, _pos)

#define TECA_NUM_OPERANDS_ERROR(_op, _nreq, _ngive, _expr, _pos)        \
    TECA_PARSER_ERROR("Operation \"" << _op                             \
         << "\" requires " << _nreq << " operands, given "              \
         << _ngive << ". ", _expr, _pos)

namespace teca_parser
{
/**
class that recognizes and extracts tokens during parsing.
given a pointer (first argument) the methods return the
number of chars in the token, or 0 when the pointer doesn't
point to a valid token, and copies the token into the buffer
(second argument).
*/
class tokenizer
{
public:
    static unsigned int get_open_group(const char *s, char *g);
    static unsigned int get_close_group(const char *s, char *g);
    static unsigned int get_constant_name(const char *s, char *c);
    static unsigned int get_variable_name(const char *s, char *v);
    static unsigned int get_unary_operator_name(const char *expr, char *op_name);
    static unsigned int get_binary_operator_name(const char *expr, char *op_name);
    static unsigned int get_ternary_operator_name(const char *expr, char *op_name);
    static unsigned int get_operator_precedence(char *op);
};

/**
convert infix expression to postfix. returns the postfix form
of the expression in a string allocated with malloc. caller to
free the string. return nullptr if there is an error.

template types implement detection of classes of syntactical
tokens. groups, constants, variables, and operators.
*/
template<typename tokenizer_t=teca_parser::tokenizer>
char *infix_to_postfix(const char *iexpr, std::set<std::string> *variables)
{
    std::vector<char*> operator_stack;
    std::vector<unsigned int> group_position; // position of un-matched open group

    const char *expr = iexpr;
    unsigned int n = strlen(expr);
    char *rexpr = static_cast<char*>(malloc(3*n));
    char *rpnexpr = rexpr;

    unsigned int token_len;
    char token[256];

    while (*expr)
    {
        // skip white space
        if (isspace(*expr))
        {
            while(*expr && isspace(*expr)) ++expr;
        }
        // recurse into grouped expression
        else if ((token_len = tokenizer_t::get_open_group(expr, token)))
        {
            token[token_len-1] = '\0';

            char *tmp = infix_to_postfix<tokenizer_t>(token+1, variables);
            unsigned int tmp_len = strlen(tmp);

            memcpy(rexpr, tmp, tmp_len);
            rexpr += tmp_len;

            expr += token_len;

            free(tmp);
        }
        // pass constants through
        else if ((token_len = tokenizer_t::get_constant_name(expr, token)))
        {
            memcpy(rexpr, token, token_len);
            rexpr += token_len;
            *rexpr++ = ' ';

            expr += token_len;
        }
        // pass variable names through, save the variable name
        else if ((token_len = tokenizer_t::get_variable_name(expr, token)))
        {
            if (variables)
                variables->insert(token);

            memcpy(rexpr, token, token_len);
            rexpr += token_len;
            *rexpr++ = ' ';

            expr += token_len;
        }
        // push operator names onto the stack
        else if ((token_len = tokenizer_t::get_ternary_operator_name(expr, token))
            || (token_len = tokenizer_t::get_binary_operator_name(expr, token))
            || (token_len = tokenizer_t::get_unary_operator_name(expr, token)))
        {
            // apply precedence rules. operators of higher precedence
            // are popped and applied
            unsigned int p1 = tokenizer_t::get_operator_precedence(token);
            while (operator_stack.size() &&
                (tokenizer_t::get_operator_precedence(operator_stack.back()) >= p1))
            {
                char *op_name = operator_stack.back();
                unsigned int op_len = strlen(op_name);

                operator_stack.pop_back();

                memcpy(rexpr, op_name, op_len);
                rexpr += op_len;
                *rexpr++ = ' ';

                free(op_name);
            }

            // push the new operator
            operator_stack.push_back(strdup(token));
            expr += token_len;
        }
        // every other input indicates an error
        else
        {
            TECA_SYNTAX_ERROR(iexpr, expr-iexpr)
            return nullptr;
        }
    }

    // catch unmatched open group
    if (group_position.size())
    {
        TECA_SYNTAX_ERROR(iexpr, group_position.back())
        return nullptr;
    }

    // apply the remaining operators
    while (operator_stack.size())
    {
        char *op_name = operator_stack.back();
        unsigned int op_len = strlen(op_name);

        operator_stack.pop_back();

        memcpy(rexpr, op_name, op_len);
        rexpr += op_len;
        *rexpr++ = ' ';

        free(op_name);
    }

    // null terminate the transformed expression
    *rexpr = '\0';
    return rpnexpr;
}

/**
evaluate a postfix expression. returns non zero if an error occurred.
the result of the evaluted expression is returned in iexpr_result.

template types define the intermediate types used in the calculation.
arg_t would likely be the const form of work_t. resolvers for constants,
variables, and operators are passed. The purpose of the resolvers is
to identify token class and implement variable lookup, and operator
evaluation.
*/
template<typename work_t, typename arg_t, typename operand_resolver_t,
typename operator_resolver_t, typename tokenizer_t=teca_parser::tokenizer>
int eval_postfix(arg_t &iexpr_result,
    const char *iexpr, operand_resolver_t &operands)
{
    if (!iexpr)
        return -1;

    char token[256];
    unsigned int token_len;

    std::vector<arg_t> var_stack;

    const char *expr = iexpr;
    while (*expr)
    {
        // skip white space
        if (isspace(*expr))
        {
            while(*expr && isspace(*expr)) ++expr;
        }
        // push constants onto the stack
        else if ((token_len = tokenizer_t::get_constant_name(expr, token)))
        {
            work_t var;
            if (operands.get_constant(token, var))
            {
                TECA_NAME_RESOLUTION_ERROR(token, iexpr, expr-iexpr)
                return -1;
            }
            var_stack.push_back(var);
            expr += token_len;
        }
        // push variables onto the stack
        else if ((token_len = tokenizer_t::get_variable_name(expr, token)))
        {
            arg_t var;
            if (operands.get_variable(token, var))
            {
                TECA_NAME_RESOLUTION_ERROR(token, iexpr, expr-iexpr)
                return -1;
            }
            var_stack.push_back(var);
            expr += token_len;
        }
        // pop 3 operands and apply ternary operators, push the result
        else if ((token_len = tokenizer_t::get_ternary_operator_name(expr, token)))
        {
            // there must be at least 3 operands
            unsigned int n_operands = var_stack.size();
            if (n_operands < 3)
            {
                TECA_NUM_OPERANDS_ERROR(token, 3, n_operands, iexpr, expr-iexpr)
                return -1;
            }

            // get the operands
            arg_t arg3 = var_stack.back();
            var_stack.pop_back();

            arg_t arg2 = var_stack.back();
            var_stack.pop_back();

            arg_t arg1 = var_stack.back();
            var_stack.pop_back();

            // invoke ternary operator
            int err_code;
            work_t result;
            if ((err_code = operator_resolver_t::invoke(token, result, arg1, arg2, arg3)))
            {
                if (err_code == -1)
                {
                    TECA_INVALID_OPERATION_ERROR(token, iexpr, expr-iexpr)
                }
                else if (err_code == -2)
                {
                    TECA_OPERATION_FAILED_ERROR(token, iexpr, expr-iexpr)
                }
                return -1;
            }

            // push result
            var_stack.push_back(result);

            expr += token_len;
        }
        // pop 2 operands and apply binary operators, push the result
        else if ((token_len = tokenizer_t::get_binary_operator_name(expr, token)))
        {
            // there must be at least 2 operands
            unsigned int n_operands = var_stack.size();
            if (n_operands < 2)
            {
                TECA_NUM_OPERANDS_ERROR(token, 2, n_operands, iexpr, expr-iexpr)
                return -1;
            }

            // get the operands
            arg_t right_arg = var_stack.back();
            var_stack.pop_back();

            arg_t left_arg = var_stack.back();
            var_stack.pop_back();

            // invoke binary operator
            int err_code;
            work_t result;
            if ((err_code = operator_resolver_t::invoke(token, result, left_arg, right_arg)))
            {
                if (err_code == -1)
                {
                    TECA_INVALID_OPERATION_ERROR(token, iexpr, expr-iexpr)
                }
                else if (err_code == -2)
                {
                    TECA_OPERATION_FAILED_ERROR(token, iexpr, expr-iexpr)
                }
                return -1;
            }

            // push result
            var_stack.push_back(result);

            expr += token_len;
        }
        // pop one operand, apply unary operator, push the result
        else if ((token_len = tokenizer_t::get_unary_operator_name(expr, token)))
        {
            // there must be at least 1 operands
            unsigned int n_operands = var_stack.size();
            if (n_operands < 1)
            {
                TECA_NUM_OPERANDS_ERROR(token, 1, n_operands, iexpr, expr-iexpr)
                return -1;
            }

            // get the operands
            arg_t arg = var_stack.back();
            var_stack.pop_back();

            // invoke unary operator
            int err_code;
            work_t result;
            if ((err_code = operator_resolver_t::invoke(token, result, arg)))
            {
                if (err_code == -1)
                {
                    TECA_INVALID_OPERATION_ERROR(token, iexpr, expr-iexpr)
                }
                else if (err_code == -2)
                {
                    TECA_OPERATION_FAILED_ERROR(token, iexpr, expr-iexpr)
                }
                return -1;
            }

            // store the result
            var_stack.push_back(result);

            // move to operator_resolver
            expr += token_len;
        }
        // the expression contains characters that are of an unknown
        // class, not constant, nor variable, nor operator
        else
        {
            TECA_SYNTAX_ERROR(iexpr, expr-iexpr)
            return -1;
        }
    }

    // the result should be on the stack, and it should be the only thing
    // on the stack
    if (var_stack.size() != 1)
    {
        TECA_SYNTAX_ERROR(iexpr, expr-iexpr)
        return -1;
    }
    iexpr_result = var_stack.back();

    return 0;
}
};

#endif
