#ifndef teca_metadata_util_h
#define teca_metadata_util_h

#include <string>
#include <set>

namespace teca_metadata_util
{
// remove post-fix from the arrays in get_upstream_request if
// the post-fix is set. For example if post-fix is set to "_filtered"
// then we remove all the variables in the "arrays" set that end with 
// this post-fix, and replace it with the actual requested array.
void remove_post_fix(std::set<std::string> &arrays, std::string post_fix);
};
#endif
