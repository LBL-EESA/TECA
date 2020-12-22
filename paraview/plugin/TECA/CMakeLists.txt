message(STATUS "==== vtk_add_module TECA ====")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:teca_core,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_data,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_io,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_alg,INTERFACE_INCLUDE_DIRECTORIES>)

set(plugin_cxx
    vtkTECATCCandidateTableReader.cxx
    vtkTECATCTrackTableReader.cxx
    vtkTECACF2Reader.cxx)

set(plugin_h
    vtkTECATCCandidateTableReader.h
    vtkTECATCTrackTableReader.h
    vtkTECACF2Reader.h)

set(plugin_xml
    TECASMReaders.xml)

if (TECA_HAS_UDUNITS)
    list(APPEND plugin_cxx vtkTECATimeAnnotation.cxx)
    list(APPEND plugin_h vtkTECATimeAnnotation.h)
    list(APPEND plugin_xml TECASMFilters.xml)
endif()

vtk_module_add_module(
    TECA
    SOURCES ${plugin_cxx}
    HEADERS ${plugin_h}
    )

vtk_module_link(TECA PUBLIC teca_alg teca_io teca_data teca_core)

paraview_add_server_manager_xmls(XMLS ${plugin_xml})
