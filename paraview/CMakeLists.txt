if (TECA_HAS_PARAVIEW)

    set(_paraview_build_plugin PVTECA)

    paraview_plugin_scan(
      PLUGIN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/plugin/paraview.plugin"
      PROVIDES_PLUGINS plugins
      REQUIRES_MODULES required_modules
      ENABLE_BY_DEFAULT ON
      )

    message(STATUS "==== found plugin ${plugins} ====")
    foreach (module IN LISTS required_modules)
      message(STATUS "checking ${module} ... ")
      if (NOT TARGET "${module}")
        message(FATAL_ERROR "Missing required module: ${module} for ${plugins}")
      endif ()
      message(STATUS "checking ${module} ... OK")
    endforeach ()

    paraview_plugin_build(PLUGINS ${plugins})

endif()
