#include "teca_priority_queue.h"
#include "teca_common.h"

#include <stdlib.h>
#include <iostream>
#include <vector>

// using contiguous ordinal keys
// randomly generate num_vals priorities, construct the queue, and update the
// priorities num_vals/2 times. finally verify that priorities are in sorted
// order.
int test_contiguous(int num_vals)
{
    // generate some values to prioritize
    std::vector<int> vals;
    for (int i = 0; i < num_vals; ++i)
    {
        int val = rand() % num_vals;
        vals.push_back(val);
    }

    std::cerr << "vals=";
    for (int i = 0; i < num_vals; ++i)
        std::cerr << vals[i] << " ";
    std::cerr << std::endl;

    /*std::function<int(int)> lookup = [&](int i) -> int
        { return vals[i]; };*/

    contiguous_key_priority<int,int> lookup(vals);
    auto q = teca_priority_queue<int, decltype(lookup)>::New(lookup);

    // test push keys
    for (int i = 0; i < num_vals; ++i)
        q->push(i);

    std::cerr << "initial state  " << std::endl
        << q << std::endl;

    // test updating priority of any element
    for (int i = 0; i < num_vals/2; ++i)
    {
        int j = rand() % num_vals;
        int vj = vals[j];
        int dvj = num_vals * (i % 2 == 0 ? -1 : 1);

        vals[j] += dvj;

        q->modified(j);

        std::cerr << "after vals[" << j << "] = " << vj
            << " + " << dvj << " = " << vals[j] << std::endl
            << q << std::endl;
    }

    // test pop
    int cur = 0;
    int prev = vals[q->peak()];
    std::cerr << "sorted = ";
    while (!q->empty())
    {
        cur = vals[q->pop()];
        std::cerr << cur << " ";
        if (prev > cur)
        {
            std::cerr << std::endl;
            TECA_ERROR(
                << "ERROR: heap ordering is violated! "
                << prev << " > " << cur)
            return -1;
        }
        prev = cur;
    }
    std::cerr << std::endl;

    return 0;
}

// using unordered keys
// randomly generate num_vals priorities, construct the queue, and update the
// priorities num_vals/2 times. finally verify that priorities are in sorted
// order.
int test_mapped(int num_vals)
{
    using map_t = std::map<int,int>;
    using map_it_t = map_t::iterator;

    // generate some values to prioritize
    map_t vals;
    for (int i = 0; i < num_vals; ++i)
    {
        int key = 3*i;
        int val = rand() % num_vals;
        vals[key] = val;
    }

    std::cerr << "vals=";
    for (map_it_t it = vals.begin(); it != vals.end(); ++it)
        std::cerr << "(key = " << it->first << ", value = " << it->second << ") ";
    std::cerr << std::endl;

    mapped_key_priority<int,int> lookup(vals);
    auto q = teca_priority_queue<int, decltype(lookup),
        std::greater<>, std::map<int, unsigned long>>::New(lookup);

    // test push keys
    std::cerr << "vals=";
    for (map_it_t it = vals.begin(); it != vals.end(); ++it)
        q->push(it->first);

    std::cerr << "initial state  " << std::endl
        << q << std::endl;

    // test updating priority of any element
    for (int i = 0; i < num_vals/2; ++i)
    {
        int j = 3*(rand() % num_vals);
        int vj = vals[j];
        int dvj = num_vals * (i % 2 == 0 ? -1 : 1);

        vals[j] += dvj;

        q->modified(j);

        std::cerr << "after vals[" << j << "] = " << vj
            << " + " << dvj << " = " << vals[j] << std::endl
            << q << std::endl;
    }

    // test pop
    int cur = 0;
    int prev = vals[q->peak()];
    std::cerr << "sorted = ";
    while (!q->empty())
    {
        cur = vals[q->pop()];
        std::cerr << cur << " ";
        if (prev < cur)
        {
            std::cerr << std::endl;
            TECA_ERROR(
                << "Heap ordering is violated! "
                << prev << " < " << cur)
            return -1;
        }
        prev = cur;
    }
    std::cerr << std::endl;

    return 0;
}

// run the test
int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "usage: a.out [num vals] [rng seed]" << std::endl;
        return -1;
    }


    int num_vals = atoi(argv[1]);
    int seed = atoi(argv[2]);

    srand(seed);

    std::cerr
        << "============================================" << std::endl
        << "Test contiguous keys" << std::endl
        << "============================================" << std::endl;

    if (test_contiguous(num_vals))
    {
        TECA_ERROR("Test contiguous failed")
        return -1;
    }

    std::cerr
        << "============================================" << std::endl
        << "Test mapped keys" << std::endl
        << "============================================" << std::endl;

    if (test_mapped(num_vals))
    {
        TECA_ERROR("Test mapped failed")
        return -1;
    }

    return 0;
}


