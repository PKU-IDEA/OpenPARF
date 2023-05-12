# raw_instance_area
set(OP_NAME raw_instance_area)

file(GLOB CPP_SOURCES
  src/raw_instance_area.cpp
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

#if(TORCH_ENABLE_CUDA)
#  add_pytorch_extension(${TARGET_NAME}_cuda ${CUDA_SOURCES}
#    EXTRA_INCLUDE_DIRS ${INCLUDE_DIRS}
#    EXTRA_LINK_LIBRARIES ${LINK_LIBS})
#    install(TARGETS ${TARGET_NAME}_cuda DESTINATION openparf/ops/${OP_NAME})
#endif(TORCH_ENABLE_CUDA)

file(GLOB INSTALL_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.py)
install(FILES ${INSTALL_SRCS} DESTINATION openparf/ops/${OP_NAME})
