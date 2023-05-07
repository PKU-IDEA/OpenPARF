option(BUILD_GMOCK "" OFF)
option(INSTALL_GTEST "" OFF)
add_subdirectory(googletest)

#add_subdirectory(pybind11)

set(BLEND2D_STATIC TRUE)
set(BLEND2D_BUILD_NO_JIT TRUE)
set(BLEND2D_NO_INSTALL TRUE)
add_subdirectory(blend2d)
set_target_properties(blend2d PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

set(INSTALL_LEMON FALSE)
add_subdirectory(lemon)
set_target_properties(lemon PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

add_subdirectory(pugixml)
add_subdirectory(yaml-cpp)