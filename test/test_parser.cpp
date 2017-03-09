#include "teca_parser.h"
#include "teca_coordinate_util.h"

#include <iostream>
using namespace std;

class scalar_operator_resolver
{
public:
    template<typename ret_t, typename arg_t>
    static ret_t or_op(const arg_t &a1, const arg_t &a2)
    {
        return ret_t(a1 || a2);
    }

    template<typename ret_t, typename arg_t>
    static ret_t and_op(const arg_t &a1, const arg_t &a2)
    {
        return ret_t(a1 && a2);
    }

    template<typename ret_t, typename arg_t>
    static ret_t not_op(const arg_t &a1)
    {
        return ret_t(!a1);
    }

    template<typename ret_t, typename arg_t>
    static ret_t less_op(const arg_t &a1, const arg_t &a2)
    {
        return ret_t(a1 < a2);
    }

    template<typename ret_t, typename arg_t>
    static ret_t leq_op(const arg_t &a1, const arg_t &a2)
    {
        return ret_t(a1 < a2);
    }

    template<typename ret_t, typename arg_t>
    static ret_t greater_op(const arg_t &a1, const arg_t &a2)
    {
        return ret_t(a1 > a2);
    }

    template<typename ret_t, typename arg_t>
    static ret_t geq_op(const arg_t &a1, const arg_t &a2)
    {
        return ret_t(a1 >= a2);
    }

    template<typename ret_t, typename arg_t>
    static ret_t eq_op(const arg_t &a1, const arg_t &a2)
    {
        return ret_t(a1 == a2);
    }

    template<typename ret_t, typename arg_t>
    static ret_t neq_op(const arg_t &a1, const arg_t &a2)
    {
        return ret_t(a1 == a2);
    }

    template<typename ret_t, typename arg_t>
    static ret_t add_op(const arg_t &a1, const arg_t &a2)
    {
        return ret_t(a1 + a2);
    }

    template<typename ret_t, typename arg_t>
    static ret_t subtract_op(const arg_t &a1, const arg_t &a2)
    {
        return ret_t(a1 - a2);
    }

    template<typename ret_t, typename arg_t>
    static ret_t times_op(const arg_t &a1, const arg_t &a2)
    {
        return ret_t(a1 * a2);
    }

    template<typename ret_t, typename arg_t>
    static ret_t divide_op(const arg_t &a1, const arg_t &a2)
    {
        return ret_t(a1 / a2);
    }

    template<typename ret_t, typename arg_t>
    static ret_t power_op(const arg_t &a1, const arg_t &a2)
    {
        return ret_t(std::pow(a1,a2));
    }

    template<typename ret_t, typename arg_t>
    static ret_t modulo_op(const arg_t &a1, const arg_t &a2)
    {
        return ret_t(a1 % a2);
    }

    template<typename ret_t, typename arg_t>
    static ret_t tcond_op(const arg_t &a1, const arg_t &a2, const arg_t &a3)
    {
        return ret_t(a1 ? a2 : a3);
    }

    template <typename ret_t, typename arg_t>
    static int invoke(const char *op, ret_t &r,
        const arg_t &a1, const arg_t &a2, const arg_t &a3)
    {
        switch (*op)
        {
            case '?': r = tcond_op<ret_t, arg_t>(a1, a2, a3); return 0;
        }
        TECA_ERROR("ternary operation " << op << " is not implemented")
        r = ret_t();
        return -1;
    }

    template <typename ret_t, typename arg_t>
    static int invoke(const char *op, ret_t &r,
        const arg_t &a1, const arg_t &a2)
    {
        switch (*op)
        {
            case '=': r = eq_op<ret_t, arg_t>(a1, a2); return 0;
            case '!': r = neq_op<ret_t, arg_t>(a1, a2); return 0;
            case '<':
                if (op[1] == '=') r = leq_op<ret_t, arg_t>(a1, a2);
                else r = less_op<ret_t, arg_t>(a1, a2);
                return 0;
            case '>':
                if (op[1] == '=') r = geq_op<ret_t, arg_t>(a1, a2);
                else r = greater_op<ret_t, arg_t>(a1, a2);
                return 0;
            case '&': r = and_op<ret_t, arg_t>(a1, a2); return 0;
            case '|': r = or_op<ret_t, arg_t>(a1, a2); return 0;
            case '+': r = add_op<ret_t, arg_t>(a1, a2); return 0;
            case '-': r = subtract_op<ret_t, arg_t>(a1, a2); return 0;
            case '*':
                if (op[1] == '*') r = power_op<ret_t, arg_t>(a1, a2);
                else r = times_op<ret_t, arg_t>(a1, a2);
                return 0;
            case '/': r = divide_op<ret_t, arg_t>(a1, a2); return 0;
            case '%': r = modulo_op<long, long>(a1, a2); return 0;
        }
        TECA_ERROR("binary operation " << op << " is not implemented")
        r = ret_t();
        return -1;
    }

    template <typename ret_t, typename arg_t>
    static ret_t invoke(const char *op, ret_t &r, const arg_t &a1)
    {
        switch (*op)
        {
            case '!':
                r = not_op<ret_t, arg_t>(a1);
                return 0;
                break;
        }
        TECA_ERROR("unary operation " << op << " is not implemented")
        r = ret_t();
        return -1;
    }

};

class scalar_operand_resolver
{
public:
    template <typename ret_t>
    int get_constant(const char *s, ret_t &c)
    {
        c = atof(s);
        return 0;
    }

    template <typename ret_t>
    int get_variable(const char *e, ret_t &v)
    {
        if (this->variables.count(e))
        {
            v = this->variables[e];
            return 0;
        }
        return -1;
    }

    std::map<std::string, double> variables;
};

// list of expression to evaluate
const char *ifix_expr[] = {
    "1 + 2",
    "2 + 1",
    "1 - 2",
    "2 - 1",
    "2 * 3",
    "3/2",
    "2**3 + 3 * 4 - 4",
    "2**(3 + 3) * 4 - 4",
    "4.1 < 5.1",
    "4.1 <= 4.1",
    "!(4.1 < 5.1)",
    "4.1 < 3.1",
    "5.1 > 4.1",
    "5.1 >= 5.1"
    "1 && 1",
    "1 && 0",
    "1 || 0 && 1",
    "1 || (0 && 1)",
    "E && F",
    "!E && !F",
    "!E && F",
    "G**3 - 2",
    "-1.0e2 / 100",
    "1.0e-2 / -100",
    "1.0e2 / 100",
    "(3 * (4 + 5) + 27)/9",
    "((((3 * (((4) + 5)) + 27))/9))",
	"pi",
    "5 % 3",
    "3. + (G < 3. ? 1. -1.)",
    "3. + (G > 3. ? 1. -1.)",
    };

// list of expected results
double expected[] = {
    3,
    3,
    -1,
    1,
    6,
    1.5,
    16,
    252,
    1,
    0,
    0,
    0,
    1,
    0,
    0,
    1,
    1,
    0,
    0,
    1,
    6,
    -1,
    -0.0001,
    1,
    6,
	6,
	3.1415,
    2,
    4,
    2
	};



int main(int, char**)
{
    scalar_operand_resolver operand_res;
    operand_res.variables["E"] = 0;
    operand_res.variables["F"] = 1;
    operand_res.variables["G"] = 2;
    operand_res.variables["pi"] = 3.1415;

    int nifix = sizeof(ifix_expr)/sizeof(char*);
    for (int i = 0; i < nifix; ++i)
    {
        std::set<std::string> req_variables;

        cerr << "\"" << ifix_expr[i] << "\" --> \"";
        // convert to postfix
        char *pfix_expr = teca_parser::infix_to_postfix
            (ifix_expr[i], &req_variables);

        if (!pfix_expr)
        {
            TECA_ERROR("failed to convert from infix to postfix")
            return -1;
        }
        cerr << pfix_expr << "\" = ";

        // evaluate the expression
        double result;
        if (teca_parser::eval_postfix<double, double,
            scalar_operand_resolver, scalar_operator_resolver>
            (result, pfix_expr, operand_res))
        {
            TECA_ERROR("failed to evaluate the postfix expression")
            return -1;
        }
        cerr << result << endl;
        free(pfix_expr);

        if (!teca_coordinate_util::equal(result, expected[i], 1e-5))
        {
            TECA_ERROR("unexpected result in expression " << i)
            return -1;
        }
    }

    return 0;
}
