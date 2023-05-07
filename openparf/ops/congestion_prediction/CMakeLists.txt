#congestion_prediction
set(OP_NAME congestion_prediction)

file(GLOB CPP_SOURCES
    src/rudy.cpp
    src/rudy_kernel.cpp
 )

file(GLOB CUDA_SOURCES
  src/rudy_cuda.cpp
  src/rudy_cuda_kernel.cu
  )

set(TARGET_NAME ${OP_NAME})

set(INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${Boost_INCLUDE_DIRS}
)

set(LINK_LIBS util)

add_pytorch_extension(${TARGET_NAME}_cpp ${CPP_SOURCES}
  EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
  EXTRA_LINK_LIBRARIES ${LINK_LIBS})
install(TARGETS ${TARGET_NAME}_cpp DESTINATION openparf/ops/${OP_NAME})

if(TORCH_ENABLE_CUDA)
  add_pytorch_extension(${TARGET_NAME}_cuda ${CUDA_SOURCES}
    EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
    EXTRA_LINK_LIBRARIES ${LINK_LIBS})
  install(TARGETS ${TARGET_NAME}_cuda DESTINATION openparf/ops/${OP_NAME})
endif(TORCH_ENABLE_CUDA)

file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION openparf/ops/${OP_NAME})

install(DIRECTORY src/data/ DESTINATION openparf/ops/congestion_prediction/src/data)
install(DIRECTORY src/models/ DESTINATION openparf/ops/congestion_prediction/src/models)
install(DIRECTORY src/options/ DESTINATION openparf/ops/congestion_prediction/src/options)
install(DIRECTORY src/util/ DESTINATION openparf/ops/congestion_prediction/src/util)
if(IS_DIRECTORY checkpoints)
 install(DIRECTORY checkpoints/ DESTINATION openparf/ops/congestion_prediction/checkpoints)
endif()