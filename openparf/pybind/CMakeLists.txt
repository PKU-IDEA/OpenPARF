# binding top database
string(TOLOWER ${PROJECT_NAME} OUTPUT_NAME)
set(OP_NAME ${OUTPUT_NAME})

file(GLOB CPP_SOURCES
  *.cpp
  )

set(TARGET_NAME ${OP_NAME})

set(INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
  ${PROJECT_SOURCE_DIR}/thirdparty/blend2d/src
  ${Boost_INCLUDE_DIRS}
  )

set(LINK_LIBS database util bookshelf verilogparser blend2d)

add_pytorch_extension(${TARGET_NAME} ${CPP_SOURCES}
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES ${LINK_LIBS})

install(
  TARGETS ${TARGET_NAME} DESTINATION openparf/
  )

#if (PYTHON_EXECUTABLE)
#  set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
#  set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
#  file(GLOB SOURCES
#    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
#    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
#    )
#  set(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${OP_NAME}.stamp")
#
#  configure_file(${SETUP_PY_IN} ${SETUP_PY})
#
#  add_custom_command(OUTPUT ${OUTPUT}
#    COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} build
#    --build-temp=${CMAKE_CURRENT_BINARY_DIR}/build
#    --build-lib=${CMAKE_CURRENT_BINARY_DIR}/lib
#    -j4
#    COMMAND rm -f ${CMAKE_CURRENT_BINARY_DIR}/../*.so
#    COMMAND ln -s ${CMAKE_CURRENT_BINARY_DIR}/lib/* ${CMAKE_CURRENT_BINARY_DIR}/..
#    COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
#    DEPENDS ${SOURCES} ${SETUP_PY} database util bookshelf
#    )
#
#  add_custom_target(clean_${OP_NAME}
#    COMMAND rm -rf ${OUTPUT}
#    ${CMAKE_CURRENT_BINARY_DIR}/build
#    ${CMAKE_CURRENT_BINARY_DIR}/*.so
#    ${CMAKE_CURRENT_BINARY_DIR}/*.py
#    )
#
#  add_custom_target(${OP_NAME} ALL DEPENDS ${OUTPUT} database util bookshelf)
#
#  install(
#    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib/ DESTINATION openparf/
#    )
#endif(PYTHON_EXECUTABLE)
#
##set(HEADERS
##  ${CMAKE_CURRENT_SOURCE_DIR}/..
##  ${PROJECT_SOURCE_DIR}/thirdparty/blend2d/src
##  ${Boost_INCLUDE_DIRS})
##file(GLOB SRCS *.cpp)
##
##add_library(pybind_objs OBJECT ${SRCS})
##target_include_directories(pybind_objs
##  PUBLIC ${HEADERS}
##  PRIVATE ${PYBIND11_INCLUDE_DIR}  # from project CMakeLists.txt
##  PRIVATE ${pybind11_INCLUDE_DIR}  # from pybind11Config
##  PRIVATE ${PYTHON_INCLUDE_DIRS})
##set_target_properties(pybind_objs PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
##
###add_library(pybind_static STATIC $<TARGET_OBJECTS:pybind_objs>)
###set_target_properties(pybind_static PROPERTIES POSITION_INDEPENDENT_CODE TRUE) #
##pybind11_add_module(pybind $<TARGET_OBJECTS:pybind_objs>)
###add_library(pybind SHARED $<TARGET_OBJECTS:pybind_objs>)
##target_include_directories(pybind PUBLIC ${HEADERS})
##target_link_libraries(pybind PUBLIC database util bookshelf Blend2D::Blend2D)
##string(TOLOWER ${PROJECT_NAME} OUTPUT_NAME)
##set_target_properties(pybind PROPERTIES OUTPUT_NAME
##  ../${OUTPUT_NAME})
##
#### The prefix and extension are provided by FindPythonLibsNew.cmake
###set_target_properties(pybind PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}")
###set_target_properties(pybind PROPERTIES SUFFIX "${PYTHON_MODULE_EXTENSION}")
###
#### -fvisibility=hidden is required to allow multiple modules compiled against
#### different pybind versions to work properly, and for some features (e.g.
#### py::module_local).  We force it on everything inside the `pybind11`
#### namespace; also turning it on for a pybind module compilation here avoids
#### potential warnings or issues from having mixed hidden/non-hidden types.
###set_target_properties(pybind PROPERTIES CXX_VISIBILITY_PRESET "hidden")
###set_target_properties(pybind PROPERTIES CUDA_VISIBILITY_PRESET "hidden")
##
##
##install(TARGETS pybind
##  DESTINATION ${CMAKE_INSTALL_PREFIX}/openparf)
