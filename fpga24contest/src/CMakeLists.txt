cmake_minimum_required(VERSION 3.5.1)

project(fpgarouter)

include_directories(.)

find_package(LAPACK REQUIRED)
find_package(ZLIB REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# in case the system has multiple thread libraries 
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

find_package(OpenMP REQUIRED)

# set(BOOST_ROOT "/home/jiaruiwang/boost-install")
find_package(Boost COMPONENTS filesystem program_options REQUIRED)
# find_package(lemon REQUIRED)
# find_package(GUROBI REQUIRED)
find_package(CapnProto 0.10.2 REQUIRED)
# find_package(ZLIB REQUIRED)

set (PATH_SRC ${CMAKE_CURRENT_SOURCE_DIR})
file (GLOB_RECURSE SOURCE_FILE ${PATH_SRC}/*.cpp ${PATH_SRC}/*.c ${PATH_SRC}/*.c++)

set (CMAKE_CXX_FLAGS "${CMKAE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")

add_executable(fpgarouter ${SOURCE_FILE})

target_include_directories(fpgarouter PRIVATE ${ZLIB_INCLUDE_DIRS} 
  /home/jiangxun/anaconda3/include
  /home/jiangxun/tools/boost/include
  ${GUROBI_INCLUDE_DIRS})
  
# add_library(gurobi_c++ STATIC IMPORTED)
# set_property(TARGET gurobi_c++ PROPERTY IMPORTED_LOCATION /home/jiangxun/gurobi951/linux64/lib/libgurobi_c++.a)
# target_link_libraries(fpgarouter gurobi_c++)

target_link_libraries(fpgarouter ${LAPACK_LIBRARIES} ${ZLIB_LIBRARIES} 
  OpenMP::OpenMP_CXX
  Threads::Threads
  ${GUROBI_LIBRARY}
  CapnProto::capnp
  )
