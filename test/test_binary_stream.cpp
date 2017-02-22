#include "teca_common.h"
#include "teca_binary_stream.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"
#include "teca_table.h"
#include "teca_programmable_algorithm.h"
#include "teca_dataset_diff.h"
#include "teca_mpi_manager.h"
#include "teca_system_interface.h"

#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#if defined(TECA_HAS_MPI)
#include <mpi.h>
#endif

using namespace std;

class random_number_generator
{
public:
    random_number_generator() : m_seed(1024) {}
    random_number_generator(int s) : m_seed(s) {}
    void init(int s) { m_seed = s; }
    int next()
    {
        int const a = 1103515245;
        int const c = 12345;
        m_seed = a * m_seed + c;
        return (m_seed >> 16) & 0x7FFF;
    }
private:
    int m_seed;
};

class teca_binary_stream_driver
{
public:
    teca_binary_stream_driver()
    { this->stress_test(); }

    void stress_test();
    void easy_test();

    void fill(teca_binary_stream &s);

    void validate(teca_binary_stream &s);

    enum {TEST_CHAR = 0, TEST_INT = 1, TEST_FLOAT = 2, TEST_DOUBLE = 3,
        TEST_STRING = 4, TEST_CHAR_ARRAY = 5, TEST_INT_ARRAY = 6,
        TEST_FLOAT_ARRAY = 7, TEST_DOUBLE_ARRAY = 8, TEST_CHAR_VECTOR = 9,
        TEST_INT_VECTOR = 10, TEST_FLOAT_VECTOR = 11, TEST_DOUBLE_VECTOR = 12,
        TEST_STRING_VECTOR = 13, TEST_CHAR_VARIANT_ARRAY = 14,
        TEST_INT_VARIANT_ARRAY = 15, TEST_FLOAT_VARIANT_ARRAY = 16,
        TEST_DOUBLE_VARIANT_ARRAY = 17, TEST_STRING_VARIANT_ARRAY = 18,
        TEST_EMPTY_VARIANT_ARRAY = 19, TEST_TABLE = 20, TEST_EMPTY_TABLE = 21,
        TEST_METADATA = 22, TEST_EMPTY_METADATA = 23,
        TEST_COUNT = 24};

private:
    int get_test_case(int i);

private:
    int m_random_order;
    int m_trip_count;
    int m_string_len;
    int m_array_len;
    int m_vector_len;
    int m_var_array_len;
    int m_table_len;
    random_number_generator m_rng;
};

int main(int argc, char **argv)
{
    teca_mpi_manager mpi_man(argc, argv);
    int rank = mpi_man.get_comm_rank();
    int n_ranks = mpi_man.get_comm_size();
    int root = n_ranks - 1;

    teca_system_interface::set_stack_trace_on_error();

    teca_binary_stream_driver driver;
    teca_binary_stream s1, s2;

    if (rank == root)
    {
        // put some data in there
        driver.fill(s1);
        // copy
        s2 = s1;
    }

    // broadcast to all ranks
    s2.broadcast(root);

    // validate the result
    driver.validate(s2);

    return 0;
}

bool same(const string &a, const string &b)
{
    if (a == b)
        return true;
    TECA_ERROR(<< b << " != " << a)
    return false;
}

bool same(double a, double b, double tol = 1e-6)
{
    double diff = std::fabs(b - a);
    if (diff <= tol)
        return true;
    TECA_ERROR("(" << b << " - " << a << " = " << diff << ") > " << tol)
    return false;
}

template <typename T>
bool same(T * a, T * b, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        if (!same(a[i], b[i]))
            return false;
    return true;
}

template <typename T>
bool same(const vector<T> &a, const vector<T> &b)
{
    size_t na = a.size();
    size_t nb = b.size();
    if (na != nb)
    {
        TECA_ERROR("vector sizes " << na << " != " << nb)
        return false;
    }
    for (size_t i = 0; i < na; ++i)
        if (!same(a[i], b[i]))
            return false;
    return true;
}

template <typename T>
bool same(p_teca_variant_array_impl<T> a, p_teca_variant_array_impl<T> b)
{
    size_t na = a->size();
    size_t nb = b->size();
    if (na != nb)
    {
        TECA_ERROR("p_teca_variant_array_impl sizes " << na << " != " << nb)
        return false;
    }
    for (size_t i = 0; i < na; ++i)
        if (!same(a->get(i), b->get(i)))
            return false;
    return true;
}

// --------------------------------------------------------------------------
void teca_binary_stream_driver::stress_test()
{
    m_random_order = 1;
    m_trip_count = 200*TEST_COUNT;
    m_string_len = 77;
    m_array_len = 187;
    m_vector_len = 177;
    m_var_array_len = 267;
    m_table_len = 157;
}

// --------------------------------------------------------------------------
void teca_binary_stream_driver::easy_test()
{
    m_random_order = 0;
    m_trip_count = TEST_COUNT;
    m_string_len = 11;
    m_array_len = 11;
    m_vector_len = 11;
    m_var_array_len = 11;
    m_table_len = 11;
}

// --------------------------------------------------------------------------
int teca_binary_stream_driver::get_test_case(int i)
{
    if (m_random_order)
        return m_rng.next() % TEST_COUNT;
    return i % TEST_COUNT;
}

// --------------------------------------------------------------------------
void teca_binary_stream_driver::fill(teca_binary_stream &s)
{
    m_rng.init(177);

#if defined(TECA_DEBUG)
    vector<int> coverage;
    if (m_random_order)
        coverage.resize(m_trip_count, 0);
#endif

    for (int i = 0; i < m_trip_count; ++i)
    {
        int act = this->get_test_case(i);

#if defined(TECA_DEBUG)
        if (m_random_order)
            coverage[act] += 1;
#endif

        switch (act)
        {
            case TEST_CHAR:
            {
                char val = 'A' + (i % 26);
#ifdef TECA_DEBUG
                cerr << "char  = " << val << endl;
#endif
                s.pack(val);
            }
            break;
            case TEST_INT:
            {
                int val = i;
#ifdef TECA_DEBUG
                cerr  << "int = " << val << endl;
#endif
                s.pack(val);
            }
            break;
            case TEST_FLOAT:
            {
                float val = i;
#ifdef TECA_DEBUG
                cerr << "float = " << val << endl;
#endif
                s.pack(val);
            }
            break;
            case TEST_DOUBLE:
            {
                double val = i;
#ifdef TECA_DEBUG
                cerr << "double = " << val << endl;
#endif
                s.pack(val);
            }
            break;
            case TEST_STRING:
            {
                string val;
                for (int j = 0; j <  m_string_len; ++j)
                    val.push_back('A' + ((i + j) % 26));
#ifdef TECA_DEBUG
                cerr << "string = " << val << endl;
#endif
                s.pack(val);
            }
            break;
            case TEST_CHAR_ARRAY:
            {
                char *val = static_cast<char*>(malloc( m_array_len));
#ifdef TECA_DEBUG
                cerr << "char array " <<  m_array_len << " = ";
#endif
                for (int j = 0; j <  m_array_len; ++j)
                {
                    val[j] = 'A' + ((i + j) % 26);
#ifdef TECA_DEBUG
                    cerr << val[j];
#endif
                }
#ifdef TECA_DEBUG
                cerr << endl;
#endif
                s.pack(val,  m_array_len);
                free(val);
            }
            break;
            case TEST_INT_ARRAY:
            {
                int *val = static_cast<int*>(malloc( m_array_len*sizeof(int)));
#ifdef TECA_DEBUG
                cerr << "int array " <<  m_array_len << " =";
#endif
                for (int j = 0; j <  m_array_len; ++j)
                {
                    val[j] = static_cast<int>(i + j);
#ifdef TECA_DEBUG
                    cerr << " " << val[j];
#endif
                }
#ifdef TECA_DEBUG
                cerr << endl;
#endif
                s.pack(val,  m_array_len);
                free(val);
            }
            break;
            case TEST_FLOAT_ARRAY:
            {
                float *val = static_cast<float*>(malloc( m_array_len*sizeof(float)));
#ifdef TECA_DEBUG
                cerr << "float array " <<  m_array_len << " =";
#endif
                for (int j = 0; j <  m_array_len; ++j)
                {
                    val[j] = static_cast<float>(i + j);
#ifdef TECA_DEBUG
                    cerr << " " << val[j];
#endif
                }
#ifdef TECA_DEBUG
                cerr << endl;
#endif
                s.pack(val,  m_array_len);
                free(val);
            }
            break;
            case TEST_DOUBLE_ARRAY:
            {
                double *val = static_cast<double*>(malloc( m_array_len*sizeof(double)));
#ifdef TECA_DEBUG
                cerr << "double array " <<  m_array_len << " =";
#endif
                for (int j = 0; j <  m_array_len; ++j)
                {
                    val[j] = static_cast<double>(i + j);
#ifdef TECA_DEBUG
                    cerr << " " << val[j];
#endif
                }
#ifdef TECA_DEBUG
                cerr << endl;
#endif
                s.pack(val,  m_array_len);
                free(val);
            }
            break;
            case TEST_CHAR_VECTOR:
            {
                vector<char> val( m_vector_len);
#ifdef TECA_DEBUG
                cerr << "char vector " <<  m_vector_len << " = ";
#endif
                for (int j = 0; j <  m_vector_len; ++j)
                {
                    val[j] = 'A' + ((i + j) % 26);
#ifdef TECA_DEBUG
                    cerr << val[j];
#endif
                }
#ifdef TECA_DEBUG
                cerr << endl;
#endif
                s.pack(val);
            }
            break;
            case TEST_INT_VECTOR:
            {
                vector<int> val( m_vector_len);
#ifdef TECA_DEBUG
                cerr << "int vector " <<  m_vector_len << " =";
#endif
                for (int j = 0; j <  m_vector_len; ++j)
                {
                    val[j] = static_cast<int>(i + j);
#ifdef TECA_DEBUG
                    cerr << " " << val[j];
#endif
                }
#ifdef TECA_DEBUG
                cerr << endl;
#endif
                s.pack(val);
            }
            break;
            case TEST_FLOAT_VECTOR:
            {
                vector<float> val( m_vector_len);
#ifdef TECA_DEBUG
                cerr << "float vector " <<  m_vector_len << " =";
#endif
                for (int j = 0; j <  m_vector_len; ++j)
                {
                    val[j] = static_cast<float>(i + j);
#ifdef TECA_DEBUG
                    cerr << " " << val[j];
#endif
                }
#ifdef TECA_DEBUG
                cerr << endl;
#endif
                s.pack(val);
            }
            break;
            case TEST_DOUBLE_VECTOR:
            {
                vector<double> val( m_vector_len);
#ifdef TECA_DEBUG
                cerr << "double vector " <<  m_vector_len << " =";
#endif
                for (int j = 0; j <  m_vector_len; ++j)
                {
                    val[j] = static_cast<double>(i + j);
#ifdef TECA_DEBUG
                    cerr << " " << val[j];
#endif
                }
#ifdef TECA_DEBUG
                cerr << endl;
#endif
                s.pack(val);
            }
            break;
            case TEST_STRING_VECTOR:
            {
                vector<string> val( m_vector_len);
#ifdef TECA_DEBUG
                cerr << "string vector " <<  m_vector_len << " =";
#endif
                for (int j = 0; j <  m_vector_len; ++j)
                {
                    string tmp;
                    for (int k = 0; k <  m_string_len; ++k)
                        tmp.push_back('A' + ((i + j) % 26));
                    val[j] = tmp;
#ifdef TECA_DEBUG
                    cerr << " " << val[j];
#endif
                }
#ifdef TECA_DEBUG
                cerr << endl;
#endif
                s.pack(val);
            }
            break;
            case TEST_CHAR_VARIANT_ARRAY:
            {
                p_teca_variant_array_impl<char> val = teca_variant_array_impl<char>::New(m_var_array_len);
#ifdef TECA_DEBUG
                cerr << "variant array char = ";
#endif
                for (int j = 0; j < m_var_array_len; ++j)
                {
                    val->set(j, 'A' + ((i + j) % 26));
#ifdef TECA_DEBUG
                    cerr << val->get(j);
#endif
                }
#ifdef TECA_DEBUG
                cerr << endl;
#endif
                val->to_stream(s);
            }
            break;
            case TEST_INT_VARIANT_ARRAY:
            {
                p_teca_variant_array_impl<int> val = teca_variant_array_impl<int>::New(m_var_array_len);
#ifdef TECA_DEBUG
                cerr << "variant array int =";
#endif
                for (int j = 0; j < m_var_array_len; ++j)
                {
                    val->set(j, static_cast<int>(i + j));
#ifdef TECA_DEBUG
                    cerr << " " << val->get(j);
#endif
                }
#ifdef TECA_DEBUG
                cerr << endl;
#endif
                val->to_stream(s);
            }
            break;
            case TEST_FLOAT_VARIANT_ARRAY:
            {
                p_teca_variant_array_impl<float> val = teca_variant_array_impl<float>::New(m_var_array_len);
#ifdef TECA_DEBUG
                cerr << "variant array float =";
#endif
                for (int j = 0; j < m_var_array_len; ++j)
                {
                    val->set(j, static_cast<float>(i + j));
#ifdef TECA_DEBUG
                    cerr << " " << val->get(j);
#endif
                }
#ifdef TECA_DEBUG
                cerr << endl;
#endif
                val->to_stream(s);
            }
            break;
            case TEST_DOUBLE_VARIANT_ARRAY:
            {
                p_teca_variant_array_impl<double> val = teca_variant_array_impl<double>::New(m_var_array_len);
#ifdef TECA_DEBUG
                cerr << "variant array double =";
#endif
                for (int j = 0; j < m_var_array_len; ++j)
                {
                    val->set(j, static_cast<double>(i + j));
#ifdef TECA_DEBUG
                    cerr << " " << val->get(j);
#endif
                }
#ifdef TECA_DEBUG
                cerr << endl;
#endif
                val->to_stream(s);
            }
            break;
            case TEST_STRING_VARIANT_ARRAY:
            {
                p_teca_variant_array_impl<string> val = teca_variant_array_impl<string>::New(m_var_array_len);
#ifdef TECA_DEBUG
                cerr << "variant array string =";
#endif
                for (int j = 0; j < m_var_array_len; ++j)
                {
                    string tmp;
                    for (int k = 0; k <  m_string_len; ++k)
                        tmp.push_back('A' + ((i + j) % 26));
                    val->set(j, tmp);
#ifdef TECA_DEBUG
                    cerr << " " << val->get(j);
#endif
                }
#ifdef TECA_DEBUG
                cerr << endl;
#endif
                val->to_stream(s);
            }
            break;
            case TEST_EMPTY_VARIANT_ARRAY:
            {
                p_teca_variant_array_impl<int> val = teca_variant_array_impl<int>::New(m_var_array_len);
#ifdef TECA_DEBUG
                cerr << "empty variant array =";
                cerr << endl;
#endif
                val->to_stream(s);
            }
            break;
            case TEST_TABLE:
            {
#ifdef TECA_DEBUG
                cerr << "table = ";
#endif
                p_teca_table table = teca_table::New();
                table->declare_columns("int", int(), "float", float(),
                    "double", double(), "string", string());
                for (int j = 0; j <  m_table_len; ++j)
                {
                    int q = i + j;
                    string tmp;
                    for (int k = 0; k <  m_string_len; ++k)
                        tmp.push_back('A' + (q % 26));
                    table << q << q << q << tmp;
                }
                table->to_stream(s);
#ifdef TECA_DEBUG
                table->to_stream(cerr);
                cerr << endl;
#endif
            }
            break;
            case TEST_EMPTY_TABLE:
            {
#ifdef TECA_DEBUG
                cerr << "table = " << endl;
#endif
                p_teca_table table = teca_table::New();
                table->to_stream(s);
            }
            break;
            case TEST_METADATA:
            {
#ifdef TECA_DEBUG
                cerr << "metadata = ";
#endif

                teca_metadata md;
                md.insert("name", string("metadata"));
                md.insert("value", 3.14);

                teca_metadata md1;
                md1.insert("id", 1);

                vector<int> ext1({0, 1, 2, 3, 4, 5});
                md1.insert("extent", ext1);

                teca_metadata md2;
                md2.insert("id", 2);

                vector<int> ext2({5, 4, 3, 2, 1, 0});
                md2.insert("extent", ext2);

                md1.insert("nest 2", md2);
                md.insert("nest 1", md1);

                md.to_stream(s);
#ifdef TECA_DEBUG
                md.to_stream(cerr);
                cerr << endl;
#endif
            }
            break;
            case TEST_EMPTY_METADATA:
            {
#ifdef TECA_DEBUG
                cerr << "empty metadata = ";
                cerr << endl;
#endif
                teca_metadata md;
                md.to_stream(s);
            }
            break;
            default:
                TECA_ERROR("Invalid case")
        }

    }

#ifdef TECA_DEBUG
    if (m_random_order)
    {
        cerr << "coverage =";
        for (int i = 0; i < TEST_COUNT; ++i)
            cerr << " " << coverage[i];
        cerr << endl;
    }
#endif
}

// --------------------------------------------------------------------------
void teca_binary_stream_driver::validate(teca_binary_stream &s)
{
    m_rng.init(177);

    for (int i = 0; i < m_trip_count; ++i)
    {
        int act = this->get_test_case(i);
        switch (act)
        {
            case TEST_CHAR:
            {
                char val;
                s.unpack(val);
#ifdef TECA_DEBUG
                cerr << "char  = " << val << endl;
#endif
                char val0 = 'A' + (i % 26);
                same(val, val0);
            }
            break;
            case TEST_INT:
            {
                int val;
                s.unpack(val);
#ifdef TECA_DEBUG
                cerr  << "int = " << val << endl;
#endif
                int val0 = i;
                same(val, val0);
            }
            break;
            case TEST_FLOAT:
            {
                float val;
                s.unpack(val);
#ifdef TECA_DEBUG
                cerr << "float = " << val << endl;
#endif
                float val0 = i;
                same(val, val0);
            }
            break;
            case TEST_DOUBLE:
            {
                double val;
                s.unpack(val);
#ifdef TECA_DEBUG
                cerr << "double = " << val << endl;
#endif
                double val0 = i;
                same(val, val0);
            }
            break;
            case TEST_STRING:
            {
                string val;
                s.unpack(val);
#ifdef TECA_DEBUG
                cerr << "string = " << val << endl;
#endif

                string val0;
                for (int j = 0; j <  m_string_len; ++j)
                    val0.push_back('A' + ((i + j) % 26));

                same(val, val0);
            }
            break;
            case TEST_CHAR_ARRAY:
            {
                char *val = static_cast<char*>(malloc( m_array_len));
                s.unpack(val,  m_array_len);

                char *val0 = static_cast<char*>(malloc( m_array_len));
#ifdef TECA_DEBUG
                cerr << "char array " <<  m_array_len << " = ";
#endif
                for (int j = 0; j <  m_array_len; ++j)
                {
                    val0[j] = 'A' + ((i + j) % 26);
#ifdef TECA_DEBUG
                    cerr << val[j];
#endif
                }
#ifdef TECA_DEBUG
                cerr << endl;
#endif

                same(val, val0,  m_array_len);

                free(val);
                free(val0);
            }
            break;
            case TEST_INT_ARRAY:
            {
                int *val = static_cast<int*>(malloc( m_array_len*sizeof(int)));
                s.unpack(val,  m_array_len);

                int *val0 = static_cast<int*>(malloc( m_array_len*sizeof(int)));
#ifdef TECA_DEBUG
                cerr << "int array " <<  m_array_len << " =";
#endif
                for (int j = 0; j <  m_array_len; ++j)
                {
                    val0[j] = static_cast<int>(i + j);
#ifdef TECA_DEBUG
                    cerr << " " << val[j];
#endif
                }
#ifdef TECA_DEBUG
                cerr << endl;
#endif

                same(val, val0,  m_array_len);

                free(val);
                free(val0);
            }
            break;
            case TEST_FLOAT_ARRAY:
            {
                float *val = static_cast<float*>(malloc( m_array_len*sizeof(float)));
                s.unpack(val,  m_array_len);

                float *val0 = static_cast<float*>(malloc( m_array_len*sizeof(float)));
#ifdef TECA_DEBUG
                cerr << "float array " <<  m_array_len << " =";
#endif
                for (int j = 0; j <  m_array_len; ++j)
                {
                    val0[j] = static_cast<float>(i + j);
#ifdef TECA_DEBUG
                    cerr << " " << val[j];
#endif
                }
#ifdef TECA_DEBUG
                cerr << endl;
#endif

                same(val, val0,  m_array_len);

                free(val);
                free(val0);
            }
            break;
            case TEST_DOUBLE_ARRAY:
            {
                double *val = static_cast<double*>(malloc( m_array_len*sizeof(double)));
                s.unpack(val,  m_array_len);

                double *val0 = static_cast<double*>(malloc( m_array_len*sizeof(double)));
#ifdef TECA_DEBUG
                cerr << "double array " <<  m_array_len << " =";
#endif
                for (int j = 0; j <  m_array_len; ++j)
                {
                    val0[j] = static_cast<double>(i + j);
#ifdef TECA_DEBUG
                    cerr << " " << val[j];
#endif
                }
#ifdef TECA_DEBUG
                cerr << endl;
#endif

                same(val, val0,  m_array_len);

                free(val);
                free(val0);
            }
            break;
            case TEST_CHAR_VECTOR:
            {
                vector<char> val;
                s.unpack(val);

                vector<char> val0( m_vector_len);
#ifdef TECA_DEBUG
                cerr << "char vector " <<  m_vector_len << " = ";
#endif
                for (int j = 0; j <  m_vector_len; ++j)
                {
                    val0[j] = 'A' + ((i + j) % 26);
#ifdef TECA_DEBUG
                    cerr << val[j];
#endif
                }
#ifdef TECA_DEBUG
                cerr << endl;
#endif

                same(val, val0);
            }
            break;
            case TEST_INT_VECTOR:
            {
                vector<int> val;
                s.unpack(val);

                vector<int> val0( m_vector_len);
#ifdef TECA_DEBUG
                cerr << "int vector " <<  m_vector_len << " =";
#endif
                for (int j = 0; j <  m_vector_len; ++j)
                {
                    val0[j] = static_cast<int>(i + j);
#ifdef TECA_DEBUG
                    cerr << " " << val[j];
#endif
                }
#ifdef TECA_DEBUG
                cerr << endl;
#endif

                same(val, val0);
            }
            break;
            case TEST_FLOAT_VECTOR:
            {
                vector<float> val;
                s.unpack(val);

                vector<float> val0( m_vector_len);
#ifdef TECA_DEBUG
                cerr << "float vector " <<  m_vector_len << " =";
#endif
                for (int j = 0; j <  m_vector_len; ++j)
                {
                    val0[j] = static_cast<float>(i + j);
#ifdef TECA_DEBUG
                    cerr << " " << val[j];
#endif
                }
#ifdef TECA_DEBUG
                cerr << endl;
#endif

                same(val, val0);
            }
            break;
            case TEST_DOUBLE_VECTOR:
            {
                vector<double> val;
                s.unpack(val);

                vector<double> val0( m_vector_len);
#ifdef TECA_DEBUG
                cerr << "double vector " <<  m_vector_len << " =";
#endif
                for (int j = 0; j <  m_vector_len; ++j)
                {
                    val0[j] = static_cast<double>(i + j);
#ifdef TECA_DEBUG
                    cerr << " " << val[j];
#endif
                }
#ifdef TECA_DEBUG
                cerr << endl;
#endif

                same(val, val0);
            }
            break;
            case TEST_STRING_VECTOR:
            {
                vector<string> val;
                s.unpack(val);

                vector<string> val0( m_vector_len);
#ifdef TECA_DEBUG
                cerr << "string vector " <<  m_vector_len << " =";
#endif
                for (int j = 0; j <  m_vector_len; ++j)
                {
                    string tmp;
                    for (int k = 0; k <  m_string_len; ++k)
                        tmp.push_back('A' + ((i + j) % 26));
                    val0[j] = tmp;
#ifdef TECA_DEBUG
                    cerr << " " << val[j];
#endif
                }
#ifdef TECA_DEBUG
                cerr << endl;
#endif

                same(val, val0);
            }
            break;
            case TEST_CHAR_VARIANT_ARRAY:
            {
                p_teca_variant_array_impl<char> val = teca_variant_array_impl<char>::New();
                val->from_stream(s);

                p_teca_variant_array_impl<char> val0 = teca_variant_array_impl<char>::New(m_var_array_len);
#ifdef TECA_DEBUG
                cerr << "variant array char = ";
#endif
                for (int j = 0; j < m_var_array_len; ++j)
                {
                    val0->set(j, 'A' + ((i + j) % 26));
#ifdef TECA_DEBUG
                    cerr << val->get(j);
#endif
                }
#ifdef TECA_DEBUG
                cerr << endl;
#endif

                same(val, val0);
            }
            break;
            case TEST_INT_VARIANT_ARRAY:
            {
                p_teca_variant_array_impl<int> val = teca_variant_array_impl<int>::New();
                val->from_stream(s);

                p_teca_variant_array_impl<int> val0= teca_variant_array_impl<int>::New(m_var_array_len);
#ifdef TECA_DEBUG
                cerr << "variant array int =";
#endif
                for (int j = 0; j < m_var_array_len; ++j)
                {
                    val0->set(j, static_cast<int>(i + j));
#ifdef TECA_DEBUG
                    cerr << " " << val->get(j);
#endif
                }
#ifdef TECA_DEBUG
                cerr << endl;
#endif

                same(val, val0);
            }
            break;
            case TEST_FLOAT_VARIANT_ARRAY:
            {
                p_teca_variant_array_impl<float> val = teca_variant_array_impl<float>::New();
                val->from_stream(s);

                p_teca_variant_array_impl<float> val0 = teca_variant_array_impl<float>::New(m_var_array_len);
#ifdef TECA_DEBUG
                cerr << "variant array float =";
#endif
                for (int j = 0; j < m_var_array_len; ++j)
                {
                    val0->set(j, static_cast<float>(i + j));
#ifdef TECA_DEBUG
                    cerr << " " << val->get(j);
#endif
                }
#ifdef TECA_DEBUG
                cerr << endl;
#endif

                same(val, val0);
            }
            break;
            case TEST_DOUBLE_VARIANT_ARRAY:
            {
                p_teca_variant_array_impl<double> val = teca_variant_array_impl<double>::New();
                val->from_stream(s);

                p_teca_variant_array_impl<double> val0 = teca_variant_array_impl<double>::New(m_var_array_len);
#ifdef TECA_DEBUG
                cerr << "variant array double =";
#endif
                for (int j = 0; j < m_var_array_len; ++j)
                {
                    val0->set(j, static_cast<double>(i + j));
#ifdef TECA_DEBUG
                    cerr << " " << val->get(j);
#endif
                }
#ifdef TECA_DEBUG
                cerr << endl;
#endif

                same(val, val0);
            }
            break;
            case TEST_STRING_VARIANT_ARRAY:
            {
                p_teca_variant_array_impl<string> val = teca_variant_array_impl<string>::New();
                val->from_stream(s);

                p_teca_variant_array_impl<string> val0 = teca_variant_array_impl<string>::New(m_var_array_len);
#ifdef TECA_DEBUG
                cerr << "variant array string =";
#endif
                for (int j = 0; j < m_var_array_len; ++j)
                {
                    string tmp;
                    for (int k = 0; k <  m_string_len; ++k)
                        tmp.push_back('A' + ((i + j) % 26));
                    val0->set(j, tmp);
#ifdef TECA_DEBUG
                    cerr << " " << val->get(j);
#endif
                }
#ifdef TECA_DEBUG
                cerr << endl;
#endif

                same(val, val0);
            }
            break;
            case TEST_EMPTY_VARIANT_ARRAY:
            {
                p_teca_variant_array_impl<int> val = teca_variant_array_impl<int>::New();
                val->from_stream(s);

#ifdef TECA_DEBUG
                cerr << "empty variant array =";
                cerr << endl;
#endif
            }
            break;
            case TEST_TABLE:
            {
                p_teca_table table = teca_table::New();
                table->from_stream(s);

#ifdef TECA_DEBUG
                cerr << "table = ";
#endif
                p_teca_table table0 = teca_table::New();
                table0->declare_columns("int", int(), "float", float(),
                    "double", double(), "string", string());
                for (int j = 0; j <  m_table_len; ++j)
                {
                    int q = i + j;
                    string tmp;
                    for (int k = 0; k <  m_string_len; ++k)
                        tmp.push_back('A' + (q % 26));
                    table0 << q << q << q << tmp;
                }
#ifdef TECA_DEBUG
                table->to_stream(cerr);
                cerr << endl;
#endif

                p_teca_programmable_algorithm base = teca_programmable_algorithm::New();
                base->set_number_of_input_connections(0);
                base->set_execute_callback([&](unsigned int,
                        const vector<const_p_teca_dataset> &,
                        const teca_metadata &) -> const_p_teca_dataset
                        { return table0; });

                p_teca_programmable_algorithm test = teca_programmable_algorithm::New();
                test->set_number_of_input_connections(0);
                test->set_execute_callback([&](unsigned int,
                        const vector<const_p_teca_dataset> &,
                        const teca_metadata &) -> const_p_teca_dataset
                        { return table; });

                p_teca_dataset_diff diff = teca_dataset_diff::New();
                diff->set_input_connection(0, base->get_output_port());
                diff->set_input_connection(1, test->get_output_port());
                diff->update();
            }
            break;
            case TEST_EMPTY_TABLE:
            {
#ifdef TECA_DEBUG
                cerr << "emptyt table = " << endl;
#endif
                p_teca_table table = teca_table::New();
                table->from_stream(s);
            }
            break;
            case TEST_METADATA:
            {
                teca_metadata md;
                md.from_stream(s);

#ifdef TECA_DEBUG
                cerr << "metadata = ";
                md.to_stream(cerr);
                cerr << endl;
#endif

                teca_metadata md0;
                md0.insert("name", string("metadata"));
                md0.insert("value", 3.14);

                teca_metadata md1;
                md1.insert("id", 1);

                vector<int> ext1({0, 1, 2, 3, 4, 5});
                md1.insert("extent", ext1);

                teca_metadata md2;
                md2.insert("id", 2);

                vector<int> ext2({5, 4, 3, 2, 1, 0});
                md2.insert("extent", ext2);

                md1.insert("nest 2", md2);
                md0.insert("nest 1", md1);

                if (!(md  == md0))
                    TECA_ERROR("tables are different")
            }
            break;
            case TEST_EMPTY_METADATA:
            {
#ifdef TECA_DEBUG
                cerr << "empty metadata = ";
                cerr << endl;
#endif
                teca_metadata md;
                md.from_stream(s);
            }
            break;
            default:
                TECA_ERROR("Invalid case")
        }

    }
}
