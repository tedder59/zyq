project(zyq)

cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 14)

set(LLVM_DIR "/home/cambricon/github/llvm-project/build/install/lib/cmake/llvm")
find_package(LLVM REQUIRED)

set(MLIR_DIR "/home/cambricon/github/llvm-project/build/install/lib/cmake/mlir")
find_package(MLIR REQUIRED)

set (MLIR_DIR_LIBS MLIRAnalysis
                   MLIRCallInterfaces
                   MLIRDialect
                   MLIRInferTypeOpInterface
                   MLIRIR
                   MLIRParser
                   MLIRPass
                   MLIRSideEffectInterfaces
                   MLIRStandard
                   MLIRTransforms)

add_definitions(-fno-rtti -fno-exceptions)

include_directories("/home/cambricon/github/llvm-project/build/install/include")
include_directories(${CMAKE_CURRENT_BINARY_DIR})

include(${LLVM_DIR}/TableGen.cmake)
include(${MLIR_DIR}/AddMLIR.cmake)

set(LLVM_TARGET_DEFINITIONS shape_inference_interface.td)
mlir_tablegen(shape_inference_interface.h.inc -gen-op-interface-decls)
mlir_tablegen(shape_inference_interface.cc.inc -gen-op-interface-defs)
add_public_tablegen_target(ShapeInferenceInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS ops.td)
mlir_tablegen(ops.h.inc -gen-op-decls)
mlir_tablegen(ops.cc.inc -gen-op-defs)
add_public_tablegen_target(OpsIncGen)

set(LLVM_TARGET_DEFINITIONS combine.td)
mlir_tablegen(combine.inc -gen-rewriters)
add_public_tablegen_target(CombineIncGen)

add_library(zyq dialect.cpp mlirgen.cpp combine.cpp
                shape_inference_pass.cpp
                lower_to_affine_loops.cpp)
add_dependencies(zyq ShapeInferenceInterfaceIncGen
                     OpsIncGen CombineIncGen)
target_link_libraries(zyq ${MLIR_DIR_LIBS})

add_executable(test test.cpp)
target_link_libraries(test zyq ${MLIR_DIR_LIBS})