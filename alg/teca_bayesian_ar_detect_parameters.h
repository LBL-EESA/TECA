#ifndef teca_bayesian_ar_detect_parameters_h
#define teca_bayesian_ar_detect_parameters_h

#include "teca_config.h"
#include "teca_algorithm.h"

TECA_SHARED_OBJECT_FORWARD_DECL(teca_bayesian_ar_detect_parameters)

/** @brief
 * An algorithm that constructs and serves up the parameter
 * table needed to run the Bayesian AR detector.
 */
class TECA_EXPORT teca_bayesian_ar_detect_parameters : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_bayesian_ar_detect_parameters)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_bayesian_ar_detect_parameters)
    TECA_ALGORITHM_CLASS_NAME(teca_bayesian_ar_detect_parameters)
    ~teca_bayesian_ar_detect_parameters();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @name row_offset
     * control the first row in the parameter table. default 0.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(long, row_offset)
    ///@}

    /** @name number_of_rows
     * control the number of rows copied into the table.  The rows are copied
     * in sequential order starting from row zero. The default value of -1 is
     * used to serve all rows. See also get_parameter_table_size.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(long, number_of_rows)
    ///@}

    /// return the number of rows in the internal parameter table.
    unsigned long get_parameter_table_size();

protected:
    teca_bayesian_ar_detect_parameters();

private:
    using teca_algorithm::get_output_metadata;

    teca_metadata get_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

    void set_modified() override;

private:
    long row_offset;
    long number_of_rows;

    struct internals_t;
    internals_t *internals;
};

#endif
