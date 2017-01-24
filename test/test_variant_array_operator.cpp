#include "teca_parser.h"
#include "teca_variant_array.h"
#include "teca_array_collection.h"
#include "teca_variant_array_operator.h"
#include "teca_variant_array_operand.h"
#include "teca_coordinate_util.h"

#include <iostream>
using namespace std;

// list of expression to evaluate
const char *ifix_expr[] = {
    "1 + 2",
    "2 + 1",
    "1 - 2",
    "2 - 1",
    "2 * 3",
    "3d/2c",
    "2d**3d + 3f * 4f - 4i",
    "2**(3 + 3) * 4 - 4",
    "4.1 < 5.1",
    "4s <= 4s",
    "!(4.1 < 5.1)",
    "4.1 < 3.1",
    "5.1 > 4.1",
    "5.1 >= 5.1"
    "1 && 1",
    "1 && 0",
    "1c || 0c && 1c",
    "1c || (0c && 1c)",
    "E && F",
    "!E && !F",
    "!E && F",
    "G**3 - 2d",
    "-1.0e2 / 100",
    "1.0e-2 / -100",
    "1.0e2 / 100",
    "(3 * (4 + 5) + 27)/9",
    "((((3 * (((4) + 5)) + 27))/9))",
	"pi",
    "5i % 3i",
    "3. + (G < 3. ? 1. -1.)",
    "3. + (G > 3. ? 1. -1.)"
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
    1,
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

const char *var_names[] = {
    "E",
    "F",
    "G",
    "pi"
    };

const char *var_values[] = {
    "0cu",
    "1cu",
    "2",
    "3.1415"
    };

int main(int, char**)
{
    using op_resolver_t = teca_variant_array_operator::resolver;
    using var_resolver_t = teca_variant_array_operand::resolver;

    var_resolver_t var_res;

    p_teca_array_collection col = teca_array_collection::New();
    var_res.set_variables(col);

    unsigned int nvars = sizeof(var_names)/sizeof(char*);
    for (unsigned int i = 0; i < nvars; ++i)
    {
        p_teca_variant_array v;
        var_res.get_constant(var_values[i], v);
        col->append(var_names[i], v);
    }

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
        const_p_teca_variant_array result;

        if (teca_parser::eval_postfix<p_teca_variant_array,
            const_p_teca_variant_array, var_resolver_t,
            op_resolver_t>(result, pfix_expr, var_res))
        {
            TECA_ERROR("failed to evaluate the postfix expression")
            return -1;
        }

        double val;
        result->get(0, val);
        cerr << val << endl;

        free(pfix_expr);

        if (!teca_coordinate_util::equal(val, expected[i], 1e-5))
        {
            TECA_ERROR("unexpected result in expression " << i)
            return -1;
        }
    }

    return 0;
}
