#ifndef teca_metadata_util_h
#define teca_metadata_util_h

#include "teca_metadata.h"

#include <string>
#include <set>

namespace teca_metadata_util
{
// given a set of names, where names end with a common string, here called
// a post-fix, modifies the set of names by removing the post fix from each
// name.
void remove_post_fix(std::set<std::string> &names, std::string post_fix);

// sets an attribute associated with the array in the attributes collection.
template <typename att_t>
void set_array_attribute(teca_metadata &md, const std::string &array_name,
    const std::string &att_name, const att_t &att_val)
{
    // get the attributes collection
    teca_metadata atts;
    md.get("attributes", atts);

    // get the existing variable attributes
    teca_metadata array_atts;
    atts.get(array_name, array_atts);

    // set the attribute
    array_atts.set(att_name, att_val);

    // update the attributes collection
    atts.set(array_name, array_atts);

    // update the metadata object
    md.set("attributes", atts);
}
};
#endif
