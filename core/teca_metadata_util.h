#ifndef teca_metadata_util_h
#define teca_metadata_util_h

#include <string>
#include <set>

namespace teca_metadata_util
{
// given a set of names, where names end with a common string, here called
// a post-fix, modifies the set of names by removing the post fix from each
// name.
void remove_post_fix(std::set<std::string> &names, std::string post_fix);

};
#endif
