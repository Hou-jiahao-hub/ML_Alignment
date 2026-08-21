cmake_minimum_required(VERSION 3.14)
project(AlignDerivLib VERSION 0.1.0 LANGUAGES CXX)

include(CTest)

find_package(Eigen3 REQUIRED NO_MODULE)

# Header-only interface library — nothing to compile, just propagate includes
add_library(AlignDeriv INTERFACE)
target_include_directories(AlignDeriv INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(AlignDeriv INTERFACE Eigen3::Eigen)
target_compile_features(AlignDeriv INTERFACE cxx_std_17)

# ── Smoke test ───────────────────────────────────────────────────────────────
option(ALIGNDERIV_BUILD_TESTS "Build smoke-test executable" ON)
if(ALIGNDERIV_BUILD_TESTS)
    add_executable(alignderiv_smoke_test tests/smoke_test.cpp)
    target_link_libraries(alignderiv_smoke_test PRIVATE AlignDeriv)
    target_compile_features(alignderiv_smoke_test PRIVATE cxx_std_17)
    add_test(NAME alignderiv_smoke COMMAND alignderiv_smoke_test)

    add_executable(alignderiv_test_jacobians tests/test_jacobians.cpp)
    target_link_libraries(alignderiv_test_jacobians PRIVATE AlignDeriv)
    target_compile_features(alignderiv_test_jacobians PRIVATE cxx_std_17)
    add_test(NAME alignderiv_jacobians COMMAND alignderiv_test_jacobians)

    add_executable(alignderiv_test_surfaces tests/test_surfaces.cpp)
    target_link_libraries(alignderiv_test_surfaces PRIVATE AlignDeriv)
    target_compile_features(alignderiv_test_surfaces PRIVATE cxx_std_17)
    add_test(NAME alignderiv_surfaces COMMAND alignderiv_test_surfaces)
endif()

# ── Python bindings via pybind11 (requires pybind11 >= 2.12 for numpy 2.x) ───
option(ALIGNDERIV_BUILD_PYTHON_PYBIND11 "Build Python bindings via pybind11" OFF)
if(ALIGNDERIV_BUILD_PYTHON_PYBIND11)
    find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(align_deriv_pb bindings/PyBindings_pybind11.cpp)
    target_link_libraries(align_deriv_pb PRIVATE AlignDeriv)
    target_compile_features(align_deriv_pb PRIVATE cxx_std_17)
endif()

# ── Python bindings via nanobind (numpy 2.x / Python 3.13 compatible) ────────
option(ALIGNDERIV_BUILD_PYTHON "Build Python bindings via nanobind" ON)
if(ALIGNDERIV_BUILD_PYTHON)
    # Resolve nanobind cmake dir from the active Python interpreter
    execute_process(
        COMMAND python3 -c "import nanobind; print(nanobind.cmake_dir())"
        OUTPUT_VARIABLE nanobind_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _nanobind_result
    )
    if(NOT _nanobind_result EQUAL 0)
        message(FATAL_ERROR "Could not find nanobind cmake dir. "
                            "Ensure nanobind is installed: pip install nanobind")
    endif()
    find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(align_deriv bindings/PyBindings.cpp)
    target_link_libraries(align_deriv PRIVATE AlignDeriv)
    target_compile_features(align_deriv PRIVATE cxx_std_17)
endif()
