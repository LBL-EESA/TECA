%{
#include "teca_config.h"
#include "teca_2d_component_area.h"
#include "teca_algorithm.h"
#include "teca_apply_binary_mask.h"
#include "teca_bayesian_ar_detect.h"
#include "teca_binary_segmentation.h"
#include "teca_cartesian_mesh_subset.h"
#include "teca_cartesian_mesh_regrid.h"
#include "teca_connected_components.h"
#include "teca_component_statistics.h"
#include "teca_2d_component_area.h"
#include "teca_component_area_filter.h"
#include "teca_latitude_damper.h"
#include "teca_dataset_diff.h"
#include "teca_dataset_capture.h"
#include "teca_dataset_source.h"
#include "teca_derived_quantity.h"
#include "teca_derived_quantity_numerics.h"
#include "teca_descriptive_statistics.h"
#include "teca_evaluate_expression.h"
#include "teca_table_region_mask.h"
#include "teca_l2_norm.h"
#include "teca_laplacian.h"
#include "teca_mask.h"
#include "teca_programmable_algorithm.h"
#include "teca_programmable_reduce.h"
#include "teca_saffir_simpson.h"
#include "teca_table_calendar.h"
#include "teca_table_sort.h"
#include "teca_table_reduce.h"
#include "teca_table_remove_rows.h"
#include "teca_table_to_stream.h"
#include "teca_tc_candidates.h"
#include "teca_tc_classify.h"
#include "teca_tc_trajectory.h"
#include "teca_tc_wind_radii.h"
#include "teca_temporal_average.h"
#include "teca_vorticity.h"
#include "teca_py_object.h"
#include "teca_py_algorithm.h"
#include "teca_py_gil_state.h"
%}

%include "teca_config.h"
%include "teca_py_common.i"
%include "teca_py_shared_ptr.i"
%include "teca_py_core.i"
%include "teca_py_data.i"
%include <std_string.i>

/***************************************************************************
 cartesian_mesh_subset
 ***************************************************************************/
%ignore teca_cartesian_mesh_subset::shared_from_this;
%shared_ptr(teca_cartesian_mesh_subset)
%ignore teca_cartesian_mesh_subset::operator=;
%include "teca_cartesian_mesh_subset.h"

/***************************************************************************
 cartesian_mesh_regrid
 ***************************************************************************/
%ignore teca_cartesian_mesh_regrid::shared_from_this;
%shared_ptr(teca_cartesian_mesh_regrid)
%ignore teca_cartesian_mesh_regrid::operator=;
%include "teca_cartesian_mesh_regrid.h"

/***************************************************************************
 connected_components
 ***************************************************************************/
%ignore teca_connected_components::shared_from_this;
%shared_ptr(teca_connected_components)
%ignore teca_connected_components::operator=;
%include "teca_connected_components.h"

/***************************************************************************
 l2_norm
 ***************************************************************************/
%ignore teca_l2_norm::shared_from_this;
%shared_ptr(teca_l2_norm)
%ignore teca_l2_norm::operator=;
%include "teca_l2_norm.h"

/***************************************************************************
 laplacian
 ***************************************************************************/
%ignore teca_laplacian::shared_from_this;
%shared_ptr(teca_laplacian)
%ignore teca_laplacian::operator=;
%include "teca_laplacian.h"


/***************************************************************************
 mask
 ***************************************************************************/
%ignore teca_mask::shared_from_this;
%shared_ptr(teca_mask)
%ignore teca_mask::operator=;
%include "teca_mask.h"

/***************************************************************************
 table_reduce
 ***************************************************************************/
%ignore teca_table_reduce::shared_from_this;
%shared_ptr(teca_table_reduce)
%ignore teca_table_reduce::operator=;
%include "teca_table_reduce.h"

/***************************************************************************
 table_to_stream
 ***************************************************************************/
%ignore teca_table_to_stream::shared_from_this;
%shared_ptr(teca_table_to_stream)
%ignore teca_table_to_stream::operator=;
%ignore teca_table_to_stream::set_stream(std::ostream *);
%include "teca_table_to_stream.h"

/***************************************************************************
 temporal_average
 ***************************************************************************/
%ignore teca_temporal_average::shared_from_this;
%shared_ptr(teca_temporal_average)
%ignore teca_temporal_average::operator=;
%include "teca_temporal_average.h"

/***************************************************************************
 vorticity
 ***************************************************************************/
%ignore teca_vorticity::shared_from_this;
%shared_ptr(teca_vorticity)
%ignore teca_vorticity::operator=;
%include "teca_vorticity.h"

/***************************************************************************
 programmable_algorithm
 ***************************************************************************/
%ignore teca_programmable_algorithm::shared_from_this;
%shared_ptr(teca_programmable_algorithm)
%extend teca_programmable_algorithm
{
    void set_report_callback(PyObject *f)
    {
        teca_py_gil_state gil;

        self->set_report_callback(teca_py_algorithm::report_callback(f));
    }

    void set_request_callback(PyObject *f)
    {
        teca_py_gil_state gil;

        self->set_request_callback(teca_py_algorithm::request_callback(f));
    }

    void set_execute_callback(PyObject *f)
    {
        teca_py_gil_state gil;

        self->set_execute_callback(teca_py_algorithm::execute_callback(f));
    }
}
%ignore teca_programmable_algorithm::operator=;
%ignore teca_programmable_algorithm::set_report_callback;
%ignore teca_programmable_algorithm::get_report_callback;
%ignore teca_programmable_algorithm::set_request_callback;
%ignore teca_programmable_algorithm::get_request_callback;
%ignore teca_programmable_algorithm::set_execute_callback;
%ignore teca_programmable_algorithm::get_execute_callback;
%include "teca_programmable_algorithm_fwd.h"
%include "teca_programmable_algorithm.h"

/***************************************************************************
 programmable_reduce
 ***************************************************************************/
%ignore teca_programmable_reduce::shared_from_this;
%shared_ptr(teca_programmable_reduce)
%extend teca_programmable_reduce
{
    void set_report_callback(PyObject *f)
    {
        teca_py_gil_state gil;

        self->set_report_callback(teca_py_algorithm::report_callback(f));
    }

    void set_request_callback(PyObject *f)
    {
        teca_py_gil_state gil;

        self->set_request_callback(teca_py_algorithm::request_callback(f));
    }

    void set_reduce_callback(PyObject *f)
    {
        teca_py_gil_state gil;

        self->set_reduce_callback(teca_py_algorithm::reduce_callback(f));
    }
}
%ignore teca_programmable_reduce::operator=;
%ignore teca_programmable_reduce::set_report_callback;
%ignore teca_programmable_reduce::get_report_callback;
%ignore teca_programmable_reduce::set_request_callback;
%ignore teca_programmable_reduce::get_request_callback;
%ignore teca_programmable_reduce::set_reduce_callback;
%ignore teca_programmable_reduce::get_reduce_callback;
%include "teca_programmable_reduce_fwd.h"
%include "teca_programmable_reduce.h"

/***************************************************************************
 derived_quantity
 ***************************************************************************/
%ignore teca_derived_quantity::shared_from_this;
%shared_ptr(teca_derived_quantity)
%extend teca_derived_quantity
{
    void set_execute_callback(PyObject *f)
    {
        teca_py_gil_state gil;

        self->set_execute_callback(teca_py_algorithm::execute_callback(f));
    }
}
%ignore teca_derived_quantity::operator=;
%ignore teca_derived_quantity::set_execute_callback;
%ignore teca_derived_quantity::get_execute_callback;
%include "teca_derived_quantity.h"
%include "teca_derived_quantity_numerics.h"

/***************************************************************************
 descriptive_statistics
 ***************************************************************************/
%ignore teca_descriptive_statistics::shared_from_this;
%shared_ptr(teca_descriptive_statistics)
%ignore teca_descriptive_statistics::operator=;
%include "teca_descriptive_statistics.h"

/***************************************************************************
 table_sort
 ***************************************************************************/
%ignore teca_table_sort::shared_from_this;
%shared_ptr(teca_table_sort)
%ignore teca_table_sort::operator=;
%include "teca_table_sort.h"

/***************************************************************************
 table_calendar
 ***************************************************************************/
%ignore teca_table_calendar::shared_from_this;
%shared_ptr(teca_table_calendar)
%ignore teca_table_calendar::operator=;
%include "teca_table_calendar.h"

/***************************************************************************
 tc_candidates
 ***************************************************************************/
%ignore teca_tc_candidates::shared_from_this;
%shared_ptr(teca_tc_candidates)
%ignore teca_tc_candidates::operator=;
%include "teca_tc_candidates.h"

/***************************************************************************
 tc_trajectory
 ***************************************************************************/
%ignore teca_tc_trajectory::shared_from_this;
%shared_ptr(teca_tc_trajectory)
%ignore teca_tc_trajectory::operator=;
%include "teca_tc_trajectory.h"

/***************************************************************************
 tc_classify
 ***************************************************************************/
%ignore teca_tc_classify::shared_from_this;
%shared_ptr(teca_tc_classify)
%ignore teca_tc_classify::operator=;
%include "teca_tc_classify.h"

/***************************************************************************
 dataset_diff
 ***************************************************************************/
%ignore teca_dataset_diff::shared_from_this;
%shared_ptr(teca_dataset_diff)
%ignore teca_dataset_diff::operator=;
%include "teca_dataset_diff.h"

/***************************************************************************
 table_region_mask
 ***************************************************************************/
%ignore teca_table_region_mask::shared_from_this;
%shared_ptr(teca_table_region_mask)
%ignore teca_table_region_mask::operator=;
%include "teca_table_region_mask.h"
%extend teca_table_region_mask
{
    TECA_PY_ALGORITHM_VECTOR_PROPERTY(unsigned long, region_size)
    TECA_PY_ALGORITHM_VECTOR_PROPERTY(unsigned long, region_start);
    TECA_PY_ALGORITHM_VECTOR_PROPERTY(double, region_x_coordinate);
    TECA_PY_ALGORITHM_VECTOR_PROPERTY(double, region_y_coordinate);
}

/***************************************************************************
 tc_activity
 ***************************************************************************/
%pythoncode %{
from teca_tc_activity import *
%}

/***************************************************************************
 tc_stats
 ***************************************************************************/
%pythoncode %{
from teca_tc_stats import *
%}

/***************************************************************************
 tc_trajectory_scalars
 ***************************************************************************/
%pythoncode %{
from teca_tc_trajectory_scalars import *
%}

/***************************************************************************
 tc_wind_radii_stats
 ***************************************************************************/
%pythoncode %{
from teca_tc_wind_radii_stats import *
%}

/***************************************************************************
 binary_segmentation
 ***************************************************************************/
%ignore teca_binary_segmentation::shared_from_this;
%shared_ptr(teca_binary_segmentation)
%ignore teca_binary_segmentation::operator=;
%include "teca_binary_segmentation.h"

/***************************************************************************
 apply_binary_mask
 ***************************************************************************/
%ignore teca_apply_binary_mask::shared_from_this;
%shared_ptr(teca_apply_binary_mask)
%ignore teca_apply_binary_mask::operator=;
%include "teca_apply_binary_mask.h"

/***************************************************************************
 Saffir-Simpson utility namespace
 ***************************************************************************/
%inline %{
struct teca_tc_saffir_simpson
{
    static int classify_mps(double w)
    { return teca_saffir_simpson::classify_mps(w); }

    static double get_lower_bound_mps(int c)
    { return teca_saffir_simpson::get_lower_bound_mps<double>(c); }

    static double get_upper_bound_mps(int c)
    { return teca_saffir_simpson::get_upper_bound_mps<double>(c); }

    static int classify_kmph(double w)
    { return teca_saffir_simpson::classify_kmph(w); }

    static double get_lower_bound_kmph(int c)
    { return teca_saffir_simpson::get_lower_bound_kmph<double>(c); }

    static double get_upper_bound_kmph(int c)
    { return teca_saffir_simpson::get_upper_bound_kmph<double>(c); }
};
%}

/***************************************************************************
 evaluate_expression
 ***************************************************************************/
%ignore teca_evaluate_expression::shared_from_this;
%shared_ptr(teca_evaluate_expression)
%ignore teca_evaluate_expression::operator=;
%include "teca_evaluate_expression.h"

/***************************************************************************
 table_remove_rows
 ***************************************************************************/
%ignore teca_table_remove_rows::shared_from_this;
%shared_ptr(teca_table_remove_rows)
%ignore teca_table_remove_rows::operator=;
%include "teca_table_remove_rows.h"

/***************************************************************************
 tc_wind_radii
 ***************************************************************************/
%ignore teca_tc_wind_radii::shared_from_this;
%shared_ptr(teca_tc_wind_radii)
%ignore teca_tc_wind_radii::operator=;
%include "teca_tc_wind_radii.h"

/***************************************************************************
 dataset_source
 ***************************************************************************/
%ignore teca_dataset_source::shared_from_this;
%shared_ptr(teca_dataset_source)
%ignore teca_dataset_source::operator=;
%include "teca_dataset_source.h"

/***************************************************************************
 dataset_capture
 ***************************************************************************/
%ignore teca_dataset_capture::shared_from_this;
%shared_ptr(teca_dataset_capture)
%ignore teca_dataset_capture::operator=;
%include "teca_dataset_capture.h"

/***************************************************************************
 2d_component_area
 ***************************************************************************/
%ignore teca_2d_component_area::shared_from_this;
%shared_ptr(teca_2d_component_area)
%ignore teca_2d_component_area::operator=;
%include "teca_2d_component_area.h"

/***************************************************************************
 latitude_damper
 ***************************************************************************/
%ignore teca_latitude_damper::shared_from_this;
%shared_ptr(teca_latitude_damper)
%ignore teca_latitude_damper::operator=;
%include "teca_latitude_damper.h"

/***************************************************************************
 bayesian_ar_detect
 ***************************************************************************/
%ignore teca_bayesian_ar_detect::shared_from_this;
%shared_ptr(teca_bayesian_ar_detect)
%ignore teca_bayesian_ar_detect::operator=;
%include "teca_bayesian_ar_detect.h"

/***************************************************************************
 component_statistics
 ***************************************************************************/
%ignore teca_component_statistics::shared_from_this;
%shared_ptr(teca_component_statistics)
%ignore teca_component_statistics::operator=;
%include "teca_component_statistics.h"

/***************************************************************************
 component_area_filter
 ***************************************************************************/
%ignore teca_component_area_filter::shared_from_this;
%shared_ptr(teca_component_area_filter)
%ignore teca_component_area_filter::operator=;
%include "teca_component_area_filter.h"
