project(teca_paraview_plugin)

if (TECA_HAS_PARAVIEW)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}
        $<TARGET_PROPERTY:teca_core,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:teca_data,INTERFACE_INCLUDE_DIRECTORIES>)

    set(plugin_srcs
        vtkTECATCCandidateTableReader.cxx
        vtkTECATCTrackTableReader.cxx
        vtkTECACF2Reader.cxx)

    set(plugin_xml
        TECASMReaders.xml)

    if (TECA_HAS_UDUNITS)
        list(APPEND plugin_srcs vtkTECATimeAnnotation.cxx)
        list(APPEND plugin_xml TECASMFilters.xml)
    endif()

    set(plugin_link teca_data teca_core teca_io teca_alg)

    include(${PARAVIEW_USE_FILE})

    ADD_PARAVIEW_PLUGIN(
        TECAParaViewPlugin "${TECA_VERSION_DESCR}"
        SERVER_MANAGER_XML ${plugin_xml}
        SERVER_MANAGER_SOURCES ${plugin_srcs}
        REQUIRED_ON_SERVER)

    target_link_libraries(TECAParaViewPlugin
        PRIVATE ${plugin_link})

endif()
