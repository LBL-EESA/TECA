#include "teca_metadata_util.h"

namespace teca_metadata_util
{
// remove post-fix from the arrays in get_upstream_request if
// the post-fix is set. For example if post-fix is set to "_filtered"
// then we remove all the variables in the "arrays" set that end with 
// this post-fix, and replace it with the actual requested array.
void remove_post_fix(std::set<std::string> &arrays, std::string post_fix)
{
    size_t postfix_len = post_fix.length();

    std::set<std::string>::iterator arrays_it;
    for (arrays_it=arrays.begin(); arrays_it!=arrays.end(); ++arrays_it)
    {
        std::string array_var = *arrays_it;
        size_t array_var_len = array_var.length();

        if (array_var_len > postfix_len)
        {
            size_t postfix_pos = array_var.find(post_fix, 
                                            array_var_len - postfix_len);
            if (postfix_pos != std::string::npos)
            {
                array_var.erase(array_var_len - postfix_len, postfix_len);

                arrays.erase(arrays_it);
                arrays.insert(array_var);
            }
        }
    }
}
};