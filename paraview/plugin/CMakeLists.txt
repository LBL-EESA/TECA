message(STATUS "==== paraview_add_plugin PVTECA ====")

paraview_add_plugin(PVTECA
  VERSION "1.0"
  MODULES TECA
  MODULE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/TECA/vtk.module"
  REQUIRED_ON_SERVER
  )
