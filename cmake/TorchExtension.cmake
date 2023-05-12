# @file TorchExtension.cmake
# @author Zizheng Guo
# @brief Use CMake to compile PyTorch extensions

add_subdirectory(thirdparty/pybind11)
find_package(OpenMP REQUIRED)
find_package(Python3 REQUIRED)
execute_process(COMMAND ${Python3_EXECUTABLE} -c 
  "import torch; print(torch.__path__[0]); print(int(torch.cuda.is_available())); print(torch.__version__);" 
  OUTPUT_VARIABLE TORCH_OUTPUT OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REPLACE "\n" ";" TORCH_OUTPUT_LIST ${TORCH_OUTPUT})
list(GET TORCH_OUTPUT_LIST 0 TORCH_INSTALL_PREFIX)
list(GET TORCH_OUTPUT_LIST 1 TORCH_ENABLE_CUDA)
list(GET TORCH_OUTPUT_LIST 2 TORCH_VERSION)
string(REPLACE "." ";" TORCH_VERSION_LIST ${TORCH_VERSION})
list(GET TORCH_VERSION_LIST 0 TORCH_MAJOR_VERSION)
list(GET TORCH_VERSION_LIST 1 TORCH_MINOR_VERSION)

message(STATUS TORCH_INSTALL_PREFIX=${TORCH_INSTALL_PREFIX})
message(STATUS TORCH_ENABLE_CUDA=${TORCH_ENABLE_CUDA})
message(STATUS TORCH_VERSION=${TORCH_MAJOR_VERSION}.${TORCH_MINOR_VERSION})

if ("${TORCH_MAJOR_VERSION}.${TORCH_MINOR_VERSION}" VERSION_LESS 1.6)
  message(SEND_ERROR "require PyTorch version greater or equal to 1.6")
endif()

if (TORCH_ENABLE_CUDA)
  find_package(CUDA 9.0 REQUIRED)
endif()

add_library(torch STATIC IMPORTED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)
find_library(TORCH_LIBRARY torch PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)
find_library(C10_LIBRARY c10 PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)
find_library(C10_CUDA_LIBRARY c10_cuda PATHS "${TORCH_INSTALL_PREFIX}/lib")
find_library(TORCH_CPU_LIBRARY torch_cpu PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)
find_library(TORCH_CUDA_LIBRARY torch_cuda PATHS "${TORCH_INSTALL_PREFIX}/lib")

if (EXISTS ${TORCH_INSTALL_PREFIX}/include)
  # torch version 1.4+
  set(TORCH_HEADER_PREFIX ${TORCH_INSTALL_PREFIX}/include)
elseif (EXISTS ${TORCH_INSTALL_PREFIX}/lib/include)
  # torch version 1.0
  set(TORCH_HEADER_PREFIX ${TORCH_INSTALL_PREFIX}/lib/include)
endif()
set(TORCH_INCLUDE_DIRS
  ${TORCH_HEADER_PREFIX}
  ${TORCH_HEADER_PREFIX}/torch/csrc/api/include)

set(LINK_LIBS ${C10_LIBRARY} ${TORCH_CPU_LIBRARY})
if (TORCH_ENABLE_CUDA)
  set(LINK_LIBS ${LINK_LIBS}
    ${C10_CUDA_LIBRARY}
    ${TORCH_CUDA_LIBRARY})
endif()

set_target_properties(torch PROPERTIES
  IMPORTED_LOCATION "${TORCH_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${TORCH_INCLUDE_DIRS}"
  INTERFACE_LINK_LIBRARIES "${LINK_LIBS}"
  INTERFACE_COMPILE_OPTIONS "-D_GLIBCXX_USE_CXX11_ABI=${CMAKE_CXX_ABI}"
  )

# CXX only 
function(add_torch_extension target_name)
  set(multiValueArgs EXTRA_INCLUDE_DIRS EXTRA_LINK_LIBRARIES EXTRA_DEFINITIONS)
  cmake_parse_arguments(ARG "" "" "${multiValueArgs}" ${ARGN})
  if (TORCH_ENABLE_CUDA)
    cuda_add_library(${target_name} STATIC ${ARG_UNPARSED_ARGUMENTS})
  else()
    # remove cuda files 
    list(FILTER ARG_UNPARSED_ARGUMENTS EXCLUDE REGEX ".*cu$" ".*cuh$")
    add_library(${target_name} STATIC ${ARG_UNPARSED_ARGUMENTS})
  endif()
  target_include_directories(${target_name} PRIVATE ${ARG_EXTRA_INCLUDE_DIRS})
  # https://stackoverflow.com/questions/12399422/how-to-set-linker-flags-for-openmp-in-cmakes-try-compile-function
  target_link_libraries(${target_name} ${ARG_EXTRA_LINK_LIBRARIES} torch pybind11::module OpenMP::OpenMP_CXX)
  target_compile_definitions(${target_name} PRIVATE 
    TORCH_EXTENSION_NAME=${target_name}
    TORCH_MAJOR_VERSION=${TORCH_MAJOR_VERSION}
    TORCH_MINOR_VERSION=${TORCH_MINOR_VERSION}
    ENABLE_CUDA=${TORCH_ENABLE_CUDA}
    ${ARG_EXTRA_DEFINITIONS})
  set_target_properties(${target_name} PROPERTIES 
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET "hidden"
    CUDA_VISIBILITY_PRESET "hidden"
    )
endfunction()

function(add_pytorch_extension target_name)
  set(multiValueArgs EXTRA_INCLUDE_DIRS EXTRA_LINK_LIBRARIES EXTRA_DEFINITIONS)
  cmake_parse_arguments(ARG "" "" "${multiValueArgs}" ${ARGN})
  #message("ARG_UNPARSED_ARGUMENTS initial = ${ARG_UNPARSED_ARGUMENTS}")
  if (TORCH_ENABLE_CUDA)
    # extract cuda files 
    set(FILTER_CUDA_SOURCES )
    foreach(ENTRY ${ARG_UNPARSED_ARGUMENTS})
      set(FILTER_CUDA_SOURCES ${FILTER_CUDA_SOURCES} ${ENTRY})
    endforeach()
    list(FILTER FILTER_CUDA_SOURCES INCLUDE REGEX ".*cu$")
    #message("FILTER_CUDA_SOURCES = ${FILTER_CUDA_SOURCES}")
    if (FILTER_CUDA_SOURCES)
      cuda_add_library(${target_name}_cuda_internal STATIC ${FILTER_CUDA_SOURCES})
      target_include_directories(${target_name}_cuda_internal PRIVATE ${ARG_EXTRA_INCLUDE_DIRS})
      target_compile_definitions(${target_name}_cuda_internal PRIVATE 
        TORCH_EXTENSION_NAME=${target_name}
        TORCH_MAJOR_VERSION=${TORCH_MAJOR_VERSION}
        TORCH_MINOR_VERSION=${TORCH_MINOR_VERSION}
        ENABLE_CUDA=${TORCH_ENABLE_CUDA}
        ${ARG_EXTRA_DEFINITIONS})
      set(INTERNAL_LINK_LIBS ${target_name}_cuda_internal)
    else()
      set(INTERNAL_LINK_LIBS )
    endif()
  else()
    set(INTERNAL_LINK_LIBS )
  endif()
  # remove cuda files 
  list(FILTER ARG_UNPARSED_ARGUMENTS EXCLUDE REGEX ".*cu$")
  #message("ARG_UNPARSED_ARGUMENTS filter = ${ARG_UNPARSED_ARGUMENTS}")
  pybind11_add_module(${target_name} MODULE ${ARG_UNPARSED_ARGUMENTS})
  target_include_directories(${target_name} PRIVATE ${ARG_EXTRA_INCLUDE_DIRS})
  target_link_libraries(${target_name} PRIVATE ${INTERNAL_LINK_LIBS} ${ARG_EXTRA_LINK_LIBRARIES} torch ${TORCH_PYTHON_LIBRARY} OpenMP::OpenMP_CXX)
  target_compile_definitions(${target_name} PRIVATE 
    TORCH_EXTENSION_NAME=${target_name}
    TORCH_MAJOR_VERSION=${TORCH_MAJOR_VERSION}
    TORCH_MINOR_VERSION=${TORCH_MINOR_VERSION}
    ENABLE_CUDA=${TORCH_ENABLE_CUDA}
    ${ARG_EXTRA_DEFINITIONS})
endfunction()
