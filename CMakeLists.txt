cmake_minimum_required(VERSION 2.8.2 FATAL_ERROR)
project(visualizer)

# In case of Makefiles if the user does not setup CMAKE_BUILD_TYPE, assume it's Release:
if (${CMAKE_GENERATOR} MATCHES ".*Makefiles")
    if("${CMAKE_BUILD_TYPE}" STREQUAL "")
        set(CMAKE_BUILD_TYPE Release)
    endif()
endif()

set  (CMAKE_CXX_STANDARD 14)

set  (project_SOURCES src/main.cpp
                      src/mainwindow.cpp
                      src/cifar_reader.cpp
                      src/linearsvm.cpp
                      src/linearsoftmax.cpp
                      src/classifier.cpp
                      src/simpleneuralnet.cpp
                      src/fisheryatesshuffle.cpp)
set  (project_HEADERS include/classifiers/mainwindow.h
                      include/classifiers/cifar_reader.h
                      include/classifiers/linearsvm.h
                      3rdparty/uni_freiburg_cv/CMatrix.h
                      3rdparty/uni_freiburg_cv/CVector.h
                      include/classifiers/linearsoftmax.h
                      include/classifiers/classifier.h
                      include/classifiers/simpleneuralnet.h
                      include/classifiers/fisheryatesshuffle.h)
set  (project_FORMS   ui/mainwindow.ui)

find_package(Eigen3 REQUIRED)
## Include gtest
# Download and unpack googletest at configure time
configure_file(cmake/CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
                 ${CMAKE_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
  include_directories("${gtest_SOURCE_DIR}/include")
endif()
## -- include gtest

find_package (Qt4 MODULE)
if(Qt4_FOUND)
  QT4_WRAP_CPP (project_HEADERS_MOC   ${project_HEADERS})
  QT4_WRAP_UI  (project_FORMS_HEADERS ${project_FORMS})
  include         (${QT_USE_FILE})
  add_definitions (${QT_DEFINITIONS})
else(Qt4_FOUND)
  find_package (Qt5Core )
  find_package (Qt5Widgets )
  find_package (Qt5Gui )
  find_package (Qt5PrintSupport )
  QT5_WRAP_CPP (project_HEADERS_MOC   ${project_HEADERS})
  QT5_WRAP_UI  (project_FORMS_HEADERS ${project_FORMS})
  set(QT_LIBRARIES Qt5::Widgets Qt5::Core Qt5::Gui)
endif(Qt4_FOUND)

include_directories(include 3rdparty ${CMAKE_CURRENT_BINARY_DIR})

add_executable  (${PROJECT_NAME} ${project_SOURCES}
                                 ${project_FORMS_HEADERS}
                                 ${project_HEADERS}
                                 ${project_HEADERS_MOC})

target_link_libraries (${PROJECT_NAME} ${QT_LIBRARIES})

# Additional targets
set(net_SOURCES src/simplenet_main.cpp
                src/simplenet_ui.cpp
                src/simpleneuralnet.cpp
                src/cifar_reader.cpp
                src/fisheryatesshuffle.cpp
                3rdparty/qcustomplot/qcustomplot.cpp)
set(net_HEADERS include/classifiers/simplenet_ui.h
                include/classifiers/simpleneuralnet.h
                include/classifiers/cifar_reader.h
                include/classifiers/fisheryatesshuffle.h
                3rdparty/qcustomplot/qcustomplot.h)
set(net_FORMS   ui/simplenet_ui.ui)

if(Qt4_FOUND)
  QT4_WRAP_CPP (net_HEADERS_MOC   ${net_HEADERS})
  QT4_WRAP_UI  (net_FORMS_HEADERS ${net_FORMS})
else(Qt4_FOUND)
  QT5_WRAP_CPP (net_HEADERS_MOC   ${net_HEADERS})
  QT5_WRAP_UI  (net_FORMS_HEADERS ${net_FORMS})
  set(QT_LIBRARIES Qt5::Widgets Qt5::Core Qt5::Gui Qt5::PrintSupport)
endif(Qt4_FOUND)

add_custom_target(copy_custom_plot ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/3rdparty/qcustomplot/qcustomplot.h ${CMAKE_CURRENT_BINARY_DIR}
)

add_executable(simple_net ${net_SOURCES} ${net_HEADERS_MOC} ${net_HEADERS} ${net_FORMS_HEADERS})
target_link_libraries (simple_net ${QT_LIBRARIES})
add_dependencies(simple_net copy_custom_plot)

add_executable(test_fisheryatesshuffle test/test_fisheryatesshuffle.cpp src/fisheryatesshuffle.cpp)
target_link_libraries(test_fisheryatesshuffle gtest_main)
add_test(NAME test_fisheryatesshuffle COMMAND test_simpleneuralnetwork)
add_executable(test_simpleneuralnetwork test/test_simpleneuralnetwork.cpp src/simpleneuralnet.cpp)
target_link_libraries(test_simpleneuralnetwork gtest_main)
add_test(NAME test_simpleneuralnetwork COMMAND test_simpleneuralnetwork)

add_executable(ensambles src/ensambles.cpp src/simpleneuralnet.cpp src/cifar_reader.cpp src/fisheryatesshuffle.cpp)
