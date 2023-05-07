cmake_minimum_required(VERSION 3.5.1)

project(fpgarouter)

# ===============================================
# Options
# ===============================================
option(CXX "enable C++ compilation" ON)

if(CXX)
  enable_language(CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

message(STATUS "C++ compilation: ${CXX}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ compiler flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "C++ compiler version: ${CMAKE_CXX_COMPILER_VERSION}")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
message(STATUS "CMake module path: ${CMAKE_MODULE_PATH}")

# ===============================================
# Dependencies
# ===============================================
find_package(LAPACK REQUIRED)
find_package(ZLIB REQUIRED)

# in case the system has multiple thread libraries
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

find_package(Boost COMPONENTS filesystem program_options REQUIRED)

find_package(GUROBI REQUIRED)

# ===============================================
# third party
# ===============================================
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)

# clipp
include(3rdParty-clipp)

# pugixml
if(NOT TARGET pugixml)
  include(3rdParty-pugixml)
else()
  set(PUGIXML_LIBRARIES pugixml)
  set(PUGIXML_INCLUDE_DIRS ${THIRD_PARTY_DIR}/pugixml)
endif()

# lemon
if(NOT TARGET lemon)
  set(INSTALL_LEMON FALSE)
  include(3rdParty-lemon)
else()
  set(LEMON_LIBRARIES lemon)
  set(LEMON_SOURCE_ROOT_DIR ${CMAKE_SOURCE_DIR}/thirdparty/lemon)
  set(LEMON_BINART_ROOT_DIR ${CMAKE_BINARY_DIR}/thirdparty/lemon)
  set(LEMON_INCLUDE_DIRS
      ${LEMON_SOURCE_ROOT_DIR}
      ${LEMON_BINART_ROOT_DIR}
  )
endif()

# mcf_solver
include(3rdParty-mcf_solver)

# taskflow
include(3rdParty-taskflow)

# ===============================================
# Targets
# ===============================================
add_subdirectory(src)
set(EXECUTABLE_NAME fpga-router)
add_executable(${EXECUTABLE_NAME} main.cpp)
target_link_libraries(${EXECUTABLE_NAME} PRIVATE ${FPGAROUTER_LIB}
)