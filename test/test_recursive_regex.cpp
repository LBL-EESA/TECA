#include "teca_file_util.h"
#include "teca_coordinate_util.h"
#include "teca_common.h"

#include <sys/stat.h>
#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>


int split_path(const std::string &path, std::vector<std::string> &comps)
{
    size_t p, n;
    std::string tmp(path);
    while (((p = tmp.rfind('/')) != std::string::npos) && ((n = tmp.size()) > 1))
    {
        comps.push_back(tmp.substr(p+1, n - p - 1));
        tmp.resize(p);
    }

    if (tmp.size())
        comps.push_back(tmp);

    return 0;
}

std::ostream &operator<<(std::ostream &os, const std::vector<std::string>  &v)
{
    for (unsigned int i = 0; i < v.size(); ++i)
        os << v[i] << ", ";
    return os;
}




int main(int argc, char **argv)
{
    if (argc != 7)
    {
        std::cerr << "test_recursive_regex [format] [calendar] "
            "[units] [t0] [t1] [nt]" << std::endl;
        return 0;
    }

    std::string format = argv[1];
    std::string calendar = argv[2];
    std::string units = argv[3];
    double t0 = std::atof(argv[4]);
    double t1 = std::atof(argv[5]);
    long nt = std::atol(argv[6]);

    double dt = (t1 - t0)/(nt - 1.0);

    for (long i = 0; i < nt; ++i)
    {
        double t = t0 + dt*i;

        std::string path;
        if (teca_coordinate_util::time_to_string(t,
            calendar, units, format, path))
        {
            TECA_ERROR("Failed to format \""  << format << "\" with calendar \""
                << calendar << "\" and units \"" << units << "\" at " << t)
            return -1;
        }

        std::vector<std::string> comps;
        split_path(path, comps);
        std::cerr << " t=" << t << " comps=" << comps << std::endl;
    }




/*
    std::vector<std::string> years = {"1990", "1991", "1992", "1993", "1994",
        "1995", "1996", "1997", "1998", "1999"};

    std::vector<std::string> months = {"January", "February", "March",
        "April", "May", "June", "July", "August", "September", "October",
        "November", "December"};

    std::vector<std::string> days = {"Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"};

    std::vector<std::string> hours = {"00:00:00", "03:00:00", "06:00:00",
        "09:00:00", "12:00:00", "15:00:00", "18:00:00", "21:00:00"};

    unsigned int n_years = years.size();
    unsigned int n_months = months.size();
    unsigned int n_days = days.size();
    unsigned int n_hours = hourse.size();

    int mode =  S_IRWXU|S_IRWXG|S_IROTH|S_IXOTH;

    for (unsigned int k = 0; k < n_years; ++k)
    {
        std::string p0 = "./" + year[k];
        if (mkdir(p0.c_str(), mode)
        {
            const char *estr = strerror(ierr);
            TECA_ERROR("Failed to create directory " << p0 << ". " << estr)
            return -1;
        }

        for (unsigned int j = 0; j < n_months; ++j)
        {
            std::string p1 = p0 + "/" + month[j];
            if (mkdir(p1.c_str(), mode)
            {
                const char *estr = strerror(ierr);
                TECA_ERROR("Failed to create directory " << p0 << ". " << estr)
                return -1;
            }

            for (unsigned int i = 0; i < n_days; ++i)
            {
                std::string p1 = p0 + "/" + month[j];
                if (mkdir(p1.c_str(), mode)
                {
                    const char *estr = strerror(ierr);
                    TECA_ERROR("Failed to create directory " << p0 << ". " << estr)
                    return -1;
                }

                for (unsigned int q = 0; q < n_hours; ++q)
                {

                }
            }
        }
    }
*/

    return 0;
}
