cmake_minimum_required(VERSION 3.20)

project(news_alpha_portfolio_agent LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_PYBIND "Build the optional pybind11 Python module" OFF)

enable_testing()

add_subdirectory(cpp)

if(BUILD_PYBIND)
  set(PYBIND11_FINDPYTHON ON)

  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(
    quant_core_py
    bindings/pybind_module.cpp
  )

  target_compile_features(
    quant_core_py
    PRIVATE cxx_std_20
  )

  target_link_libraries(
    quant_core_py
    PRIVATE quant_core
  )
endif()