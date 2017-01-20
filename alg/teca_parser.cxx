#include "teca_parser.h"

namespace teca_parser
{

// --------------------------------------------------------------------------
unsigned int tokenizer::get_constant_name(const char *s, char *c)
{
    const char *s0 = s;

    int nd = 0;
    int ne = 0;
    int nn = 0;
    int nt = 0;

    // skip leading + -, then the only time they could be encountered
    // is after E or e. NOTE: +/- at the start is problematic in
    // statements like 3+2 or 3-2, here + and - are interpeted as
    // belonging to the constant 2. one will have to use spaces
    // like 3 + 2 and 3 - 2.
    if (((s[0] == '+') || (s[0] == '-'))
        && (isdigit(s[1]) || (s[1] == '.')))
        *c++ = *s++;

    while (*s)
    {
        switch (*s)
        {
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
                *c++ = *s++;
                ++nn;
                continue;
                break;

            case 'd':
            case 'f':
            case 'L':
            case 'l':
            case 'i':
            case 's':
            case 'c':
                if (!nn) return 0; // invalid w/o preceding digit
                if (nt++) return 0; // more than one type spec
                *c++ = *s++;
                if (*s == 'u') *c++ = *s++;
                continue;
                break;

            case '.':
                if (nd++) return 0;// more than 1 .
                *c++ = *s++;
                continue;
                break;

            case 'E':
            case 'e':
                if (ne++ || !nn) return 0;// more than 1 E e
                *c++ = *s++;
                if ((*s == '+') || (*s == '-')) *c++ = *s++;
                continue;
                break;
        }
        // character does not belong to the constant stops processing
        break;
    }

    // at least one digit should have been found. this catches errors
    // like 3 + E where E is a variable, and other silly cases of illformed
    // constants
    if (!nn) return 0;

    *c = '\0';
    return s-s0;
}

// --------------------------------------------------------------------------
unsigned int tokenizer::get_variable_name(const char *s, char *v)
{
    const char *s0 = s;
    if (isdigit(*s)) return 0;
    while(*s && (isalnum(*s) || (*s == '_'))) *v++ = *s++;
    *v = '\0';
    return s-s0;
}

// --------------------------------------------------------------------------
unsigned int tokenizer::get_open_group(const char *s, char *g)
{
    const char *s0 = s;
    if (*s == '(')
    {
        int no = 1;
        *g++ = *s++;
        while (*s && no)
        {
            switch (*s)
            {
                case '(': ++no; break;
                case ')': --no; break;
            }
            *g++ = *s++;
        }
        if (no) return 0;
    }
    *g = '\0';
    return s-s0;
}

// --------------------------------------------------------------------------
unsigned int tokenizer::get_close_group(const char *s, char *g)
{
    g[0] = g[1] = '\0';
    if (*s == ')')
    {
        g[1] = *s;
        return 1;
    }
    return 0;
}

// --------------------------------------------------------------------------
unsigned int tokenizer::get_unary_operator_name(const char *expr, char *op_name)
{
    char c0 = expr[0];

    op_name[0] = c0;
    op_name[1] = '\0';

    if ((c0 == '!') || (c0 == '~'))
    {
        return 1;
    }

    return 0;
}

// --------------------------------------------------------------------------
unsigned int tokenizer::get_binary_operator_name(const char *expr, char *op_name)
{
    char c0 = expr[0];
    char c1 = expr[1];

    op_name[0] = c0;
    op_name[1] = op_name[2] = '\0';

    // <= >= == != && || **
    if (((c1 == '=') && ((c0 == '<') || (c0 == '>') || (c0 == '!') || (c0 == '=')))
        || (((c1 == '&') || (c1 == '|') || (c1 == '*')) && (c1 == c0)))
    {
        op_name[1] = c1;
        return 2;
    }
    // & | = + - * / % < >
    else if (/*(c0 == '&') || (c0 == '|') || (c0 == '=') ||*/ (c0 == '%')
        || (c0 == '+') || (c0 == '*') || (c0 == '-') || (c0 == '/')
        || (c0 == '<') || (c0 == '>'))
    {
        return 1;
    }

    return 0;
}

// --------------------------------------------------------------------------
unsigned int tokenizer::get_ternary_operator_name(const char *expr, char *op_name)
{
    char c0 = expr[0];

    op_name[0] = c0;
    op_name[1] = '\0';

    if (c0 == '?')
    {
        return 1;
    }

    return 0;
}

// --------------------------------------------------------------------------
unsigned int tokenizer::get_operator_precedence(char *op)
{
    char c0 = op[0];
    char c1 = op[1];
    switch (c0)
    {
        case '!': // ! !=
            if (c1 == '=') return 200;
            return 500;
        case '/':
        case '%':
        case '*': // * ** / %
            if (c1 == '*') return 500;
            return 400;
        case '+':
        case '-': // + -
            return 300;
        case '=':
        case '<':
        case '>':
        case '&':
        case '|': // && || == < <= > >=
            return 200;
        case '?': // ?
            return 100;
    }
    return 0;
}

};
