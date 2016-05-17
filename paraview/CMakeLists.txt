cmake_minimum_required(VERSION 2.8)
cmake_policy(SET CMP0022 NEW)

project(teca_paraview_plugin)

if (TECA_HAS_ParaView)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}
        $<TARGET_PROPERTY:teca_core,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:teca_data,INTERFACE_INCLUDE_DIRECTORIES>)

    set(plugin_srcs vtkTECATableReader.cxx)
    set(plugin_link teca_data teca_core)

    include(${PARAVIEW_USE_FILE})

    ADD_PARAVIEW_PLUGIN(
        TECAParaViewPlugin "${TECA_VERSION_DESCR}"
        SERVER_MANAGER_XML TECASMReaders.xml
        SERVER_MANAGER_SOURCES ${plugin_srcs}
        REQUIRED_ON_SERVER)

    target_link_libraries(TECAParaViewPlugin
        PRIVATE ${plugin_link})
endif()
