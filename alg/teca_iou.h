#ifndef teca_iou_h
#define teca_iou_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_iou)

/// An algorithm that computes iou from a pair of segmented fields.
class TECA_EXPORT teca_iou : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_iou)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_iou)
    TECA_ALGORITHM_CLASS_NAME(teca_iou)
    ~teca_iou();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @name iou_field_0_variable
     * set the arrays that contain the input fields to compute iou
     * from.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, iou_field_0_variable)
    ///@}

    /** @name iou_field_1_variable
     * set the arrays that contain the input fields to compute iou
     * from.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, iou_field_1_variable)
    ///@}

    /** @name fill_val_0
     * set the fill value for iou_field_0
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(double, fill_val_0)
    ///@}

    /** @name fill_val_1
     * set the fill value for iou_field_1
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(double, fill_val_1)
    ///@}

    /** @name iou_variable
     * set the name of the array to store the result in.  the default is
     * "iou"
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, iou_variable)
    ///@}

protected:
    teca_iou();

    std::string get_iou_field_0_variable(const teca_metadata &request);
    std::string get_iou_field_1_variable(const teca_metadata &request);
    double get_fill_val_0(const teca_metadata &request);
    double get_fill_val_1(const teca_metadata &request);
    std::string get_iou_variable(const teca_metadata &request);

private:
    using teca_algorithm::get_output_metadata;

    teca_metadata get_output_metadata(
        unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    std::vector<teca_metadata> get_upstream_request(
        unsigned int port,
        const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(
        unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    std::string iou_field_0_variable;
    std::string iou_field_1_variable;
    double fill_val_0;
    double fill_val_1;
    std::string iou_variable;
};

#endif
