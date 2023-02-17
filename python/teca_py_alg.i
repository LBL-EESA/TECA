%{
#include "teca_config.h"
#include "teca_2d_component_area.h"
#include "teca_algorithm.h"
#include "teca_apply_binary_mask.h"
#include "teca_apply_tempest_remap.h"
#include "teca_bayesian_ar_detect.h"
#include "teca_bayesian_ar_detect_parameters.h"
#include "teca_binary_segmentation.h"
#include "teca_cartesian_mesh_coordinate_transform.h"
#include "teca_cartesian_mesh_source.h"
#include "teca_cartesian_mesh_subset.h"
#include "teca_cartesian_mesh_regrid.h"
#include "teca_connected_components.h"
#include "teca_component_statistics.h"
#include "teca_component_area_filter.h"
#include "teca_dataset_diff.h"
#include "teca_dataset_source.h"
#include "teca_derived_quantity.h"
#include "teca_derived_quantity_numerics.h"
#include "teca_descriptive_statistics.h"
#include "teca_evaluate_expression.h"
#include "teca_elevation_mask.h"
#include "teca_integrated_vapor_transport.h"
#include "teca_indexed_dataset_cache.h"
#include "teca_l2_norm.h"
#include "teca_laplacian.h"
#include "teca_latitude_damper.h"
#include "teca_mask.h"
#include "teca_normalize_coordinates.h"
#include "teca_spatial_executive.h"
#include "teca_regional_moisture_flux.h"
#include "teca_surface_integral.h"
#include "teca_rename_variables.h"
#include "teca_saffir_simpson.h"
#include "teca_temporal_reduction.h"
#if defined(TECA_HAS_SHAPELIB)
#include "teca_shape_file_mask.h"
#endif
#include "teca_table_calendar.h"
#include "teca_table_join.h"
#include "teca_table_sort.h"
#include "teca_table_reduce.h"
#include "teca_table_region_mask.h"
#include "teca_table_remove_rows.h"
#include "teca_table_to_stream.h"
#include "teca_tc_candidates.h"
#include "teca_tc_classify.h"
#include "teca_tc_trajectory.h"
#include "teca_tc_wind_radii.h"
#include "teca_temporal_index_select.h"
#include "teca_simple_moving_average.h"
#include "teca_time_axis_convolution.h"
#include "teca_unpack_data.h"
#include "teca_valid_value_mask.h"
#include "teca_vertical_reduction.h"
#include "teca_vorticity.h"

#include "teca_py_object.h"
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
 simple_moving_average
 ***************************************************************************/
%ignore teca_simple_moving_average::shared_from_this;
%shared_ptr(teca_simple_moving_average)
%ignore teca_simple_moving_average::operator=;
%include "teca_simple_moving_average.h"

/***************************************************************************
 vorticity
 ***************************************************************************/
%ignore teca_vorticity::shared_from_this;
%shared_ptr(teca_vorticity)
%ignore teca_vorticity::operator=;
%include "teca_vorticity.h"

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

/***************************************************************************
 pytorch_algorithm
 ***************************************************************************/
#ifdef TECA_HAS_PYTORCH
%pythoncode "teca_pytorch_algorithm.py"
#endif

/***************************************************************************
 deeplab_ar_detect
 ***************************************************************************/
#ifdef TECA_HAS_PYTORCH
%pythoncode "teca_deeplab_ar_detect.py"
#endif

/***************************************************************************
 tc_activity
 ***************************************************************************/
%pythoncode "teca_tc_activity.py"

/***************************************************************************
 tc_stats
 ***************************************************************************/
%pythoncode "teca_tc_stats.py"

/***************************************************************************
 tc_trajectory_scalars
 ***************************************************************************/
%pythoncode "teca_tc_trajectory_scalars.py"

/***************************************************************************
 tc_wind_radii_stats
 ***************************************************************************/
%pythoncode "teca_tc_wind_radii_stats.py"

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
 bayesian_ar_detect_parameters
 ***************************************************************************/
%ignore teca_bayesian_ar_detect_parameters::shared_from_this;
%shared_ptr(teca_bayesian_ar_detect_parameters)
%ignore teca_bayesian_ar_detect_parameters::operator=;
%include "teca_bayesian_ar_detect_parameters.h"

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

/***************************************************************************
 normalize_coordinates
 ***************************************************************************/
%ignore teca_normalize_coordinates::shared_from_this;
%shared_ptr(teca_normalize_coordinates)
%ignore teca_normalize_coordinates::operator=;
%include "teca_normalize_coordinates.h"

/***************************************************************************
 cartesian_mesh_source
 ***************************************************************************/
%ignore teca_cartesian_mesh_source::shared_from_this;
%shared_ptr(teca_cartesian_mesh_source)
%ignore teca_cartesian_mesh_source::operator=;
%ignore operator==(const field_generator &l, const field_generator &r);
%ignore operator!=(const field_generator &l, const field_generator &r);
%include "teca_cartesian_mesh_source.h"

/***************************************************************************
 cpp_temporal_reduction
 ***************************************************************************/
%ignore teca_cpp_temporal_reduction::shared_from_this;
%shared_ptr(teca_cpp_temporal_reduction)
%ignore teca_cpp_temporal_reduction::operator=;
%include "teca_temporal_reduction.h"

/***************************************************************************
 python_temporal_reduction
 ***************************************************************************/
%pythoncode "teca_temporal_reduction.py"

/***************************************************************************
 vertical_reduction
 ***************************************************************************/
%ignore teca_vertical_reduction::shared_from_this;
%shared_ptr(teca_vertical_reduction)
%ignore teca_vertical_reduction::operator=;
%include "teca_vertical_reduction.h"

/***************************************************************************
 integrated_vapor_transport
 ***************************************************************************/
%ignore teca_integrated_vapor_transport::shared_from_this;
%shared_ptr(teca_integrated_vapor_transport)
%ignore teca_integrated_vapor_transport::operator=;
%include "teca_integrated_vapor_transport.h"

/***************************************************************************
 valid_value_mask
 ***************************************************************************/
%ignore teca_valid_value_mask::shared_from_this;
%shared_ptr(teca_valid_value_mask)
%ignore teca_valid_value_mask::operator=;
%include "teca_valid_value_mask.h"

/***************************************************************************
 unpack_data
 ***************************************************************************/
%ignore teca_unpack_data::shared_from_this;
%shared_ptr(teca_unpack_data)
%ignore teca_unpack_data::operator=;
%include "teca_unpack_data.h"

/***************************************************************************
 elevation_mask
 ***************************************************************************/
%ignore teca_elevation_mask::shared_from_this;
%shared_ptr(teca_elevation_mask)
%ignore teca_elevation_mask::operator=;
%include "teca_elevation_mask.h"

/***************************************************************************
 rename_variables
 ***************************************************************************/
%ignore teca_rename_variables::shared_from_this;
%shared_ptr(teca_rename_variables)
%ignore teca_rename_variables::operator=;
%include "teca_rename_variables.h"

/***************************************************************************
 shape_file_mask
 ***************************************************************************/
#if defined(TECA_HAS_SHAPELIB)
%ignore teca_shape_file_mask::shared_from_this;
%shared_ptr(teca_shape_file_mask)
%ignore teca_shape_file_mask::operator=;
%include "teca_shape_file_mask.h"
#endif

/***************************************************************************
 indexed_dataset_cache
 ***************************************************************************/
%ignore teca_indexed_dataset_cache::shared_from_this;
%shared_ptr(teca_indexed_dataset_cache)
%ignore teca_indexed_dataset_cache::operator=;
%include "teca_indexed_dataset_cache.h"

/***************************************************************************
 apply_tempest_remap
 ***************************************************************************/
%ignore teca_apply_tempest_remap::shared_from_this;
%shared_ptr(teca_apply_tempest_remap)
%ignore teca_apply_tempest_remap::operator=;
%include "teca_apply_tempest_remap.h"

/***************************************************************************
 potential_intensity
 ***************************************************************************/
#if defined(TECA_HAS_TCPYPI)
%pythoncode "teca_potential_intensity.py"
#endif

/***************************************************************************
 cartesian_mesh_coordinate_transform
 ***************************************************************************/
%ignore teca_cartesian_mesh_coordinate_transform::shared_from_this;
%shared_ptr(teca_cartesian_mesh_coordinate_transform)
%ignore teca_cartesian_mesh_coordinate_transform::operator=;
%include "teca_cartesian_mesh_coordinate_transform.h"

/***************************************************************************
 time_axis_convolution
 ***************************************************************************/
%ignore teca_time_axis_convolution::shared_from_this;
%shared_ptr(teca_time_axis_convolution)
%ignore teca_time_axis_convolution::operator=;
%include "teca_time_axis_convolution.h"

/***************************************************************************
 python_vertical_reduction
 ***************************************************************************/
%pythoncode "teca_python_vertical_reduction.py"

/***************************************************************************
 lapse_rate
 ***************************************************************************/
%pythoncode "teca_lapse_rate.py"

/***************************************************************************
 spatial_executive
 ***************************************************************************/
%ignore teca_spatial_executive::shared_from_this;
%shared_ptr(teca_spatial_executive)
%ignore teca_spatial_executive::operator=;
%include "teca_spatial_executive.h"

/***************************************************************************
 temporal_index_select
 ***************************************************************************/
%ignore teca_temporal_index_select::shared_from_this;
%shared_ptr(teca_temporal_index_select)
%ignore teca_temporal_index_select::operator=;
%include "teca_temporal_index_select.h"

/***************************************************************************
 temporal_percentile
 ***************************************************************************/
%pythoncode "teca_temporal_percentile.py"

/***************************************************************************
 spectral_filter
 ***************************************************************************/
%pythoncode "teca_spectral_filter.py"

/***************************************************************************
 dataset_source
 ***************************************************************************/
%ignore teca_dataset_source::shared_from_this;
%shared_ptr(teca_dataset_source)
%ignore teca_dataset_source::operator=;
%include "teca_dataset_source.h"

/***************************************************************************
 table_join
 ***************************************************************************/
%ignore teca_table_join::shared_from_this;
%shared_ptr(teca_table_join)
%ignore teca_table_join::operator=;
%include "teca_table_join.h"

/***************************************************************************
 regional_moisture_flux
 ***************************************************************************/
%ignore teca_regional_moisture_flux::shared_from_this;
%shared_ptr(teca_regional_moisture_flux)
%ignore teca_regional_moisture_flux::operator=;
%include "teca_regional_moisture_flux.h"

/***************************************************************************
 surface_integral
 ***************************************************************************/
%ignore teca_surface_integral::shared_from_this;
%shared_ptr(teca_surface_integral)
%ignore teca_surface_integral::operator=;
%include "teca_surface_integral.h"
