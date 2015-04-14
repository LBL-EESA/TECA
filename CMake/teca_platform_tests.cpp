#ifdef CXX11_REGEX_TEST
#include <regex>
#include <string>
#include <iostream>
int main(int argc, char **argv)
{
    try
    {
        std::string s("this subject has a submarine as a subsequence");
        std::regex e("\\b(sub)([^ ]*)");   // matches words beginning by "sub"
        std::smatch m;
        int n_matches = 0;
        while (std::regex_search (s,m,e))
        {
            ++n_matches;
            s = m.suffix().str();
        }
        if (n_matches == 3)
            return 1;
    }
    catch (std::regex_error &err)
    {}
    return 0;
}
#endif
