#include <iostream>
#include <sstream>
#include <cstring>

#include "teca_common.h"
#include "teca_type_select.h"

const char *expected_output = {
    "double"
    ",double"
    ",char"
    ",char"
    ",double"
    ",double"
    ",unsigned long long"
    ",unsigned long long"
    ",double"
    ",double"
    ",float"
    ",float"
    ",long"
    ",long"
    ",unsigned int"
    ",unsigned int"
    };

int main()
{
    std::ostringstream oss;
    oss << teca_type_select::elevate<double,char>::type_name() << ","
        << teca_type_select::elevate<char,double>::type_name() << ","
        << teca_type_select::decay<double,char>::type_name() << ","
        << teca_type_select::decay<char,double>::type_name() << ","
        << teca_type_select::elevate<double,unsigned long long>::type_name() << ","
        << teca_type_select::elevate<unsigned long long,double>::type_name() << ","
        << teca_type_select::decay<double,unsigned long long>::type_name() << ","
        << teca_type_select::decay<unsigned long long,double>::type_name() << ","
        << teca_type_select::elevate<double,float>::type_name() << ","
        << teca_type_select::elevate<float,double>::type_name() << ","
        << teca_type_select::decay<double,float>::type_name() << ","
        << teca_type_select::decay<float,double>::type_name() << ","
        << teca_type_select::elevate<unsigned long,int>::type_name() << ","
        << teca_type_select::elevate<int,unsigned long>::type_name() << ","
        << teca_type_select::decay<unsigned long,int>::type_name() << ","
        << teca_type_select::decay<int,unsigned long>::type_name();

    if (strcmp(expected_output, oss.str().c_str()))
    {
        TECA_ERROR("test failed." << std::endl
            << "expected: " << expected_output << std::endl
            << "got     : " << oss.str())
        return -1;
    }

    return 0;
}
