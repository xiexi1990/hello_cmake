# Set the minimum version of CMake that can be used
# To find the cmake version run
# $ cmake --version
cmake_minimum_required(VERSION 3.5)

# Set the project name
project (hello_cmake CUDA)

# find_package(ZLIB REQUIRED)

set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")

# set(cnpy_lib "/usr/local/lib/libcnpy.so")
# Add an executable
add_executable(hello_cmake hello_cmake.cu )

# target_link_libraries(${PROJECT_NAME} PUBLIC ${cnpy_lib} ${ZLIB_LIBRARIES})

# set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON CXX_STANDARD 14)
