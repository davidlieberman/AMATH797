cmake_minimum_required(VERSION 3.17)
project(proj)

set(CMAKE_CXX_STANDARD 20)

set(CONAN_CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS_RELEASE "-ltbb -mavx -mavx2 -mfma -O3 -fno-unroll-loops -pthread")
#set(CMAKE_CXX_FLAGS_RELEASE "-DMKL_ILP64 -m64 -I${MKLROOT}/include")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

#include_directories(/opt/intel/mkl/include/)
#link_directories(/opt/intel/mkl/lib/intel64 /opt/intel/lib/intel64)

add_executable(proj main.cpp)
target_link_libraries(proj ${CONAN_LIBS})
#target_link_libraries(proj -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core -ltbb -lstdc++ -lpthread -lm -ldl)
