project(teca_io)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:teca_core,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:teca_data,INTERFACE_INCLUDE_DIRECTORIES>
    )

set(teca_io_srcs
    teca_file_util.cxx
    teca_table_writer.cxx
    )

if (WIN32)
    list(APPEND teca_io_srcs win_windirent.cxx)
endif()

find_package(NetCDF)
if (NETCDF_FOUND)
    message(STATUS "NetCDF support -- enabled")
    list(APPEND teca_io_srcs teca_cf_reader.cxx)
    include_directories(${NETCDF_INCLUDE_DIRS})
else()
    message(STATUS "NetCDF support -- not found. set NETCDF_DIR to enable.")
endif()

find_package(VTK QUIET)
if (VTK_FOUND)
    message(STATUS "VTK support -- enabled")
    include(${VTK_USE_FILE})
    list(APPEND teca_io_srcs teca_vtk_cartesian_mesh_writer.cxx)
else()
    message(STATUS "VTK support -- not found. set VTK_DIR to enable.")
endif()

add_library(teca_io ${teca_io_srcs})
target_link_libraries(teca_io teca_core teca_data)

if (NETCDF_FOUND)
    target_link_libraries(teca_io ${NETCDF_LIBRARIES})
endif()

if (VTK_FOUND)
    target_link_libraries(teca_io ${VTK_LIBRARIES})
endif()

target_include_directories(teca_io
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
    )
