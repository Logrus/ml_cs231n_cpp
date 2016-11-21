cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(visualizer)

# In case of Makefiles if the user does not setup CMAKE_BUILD_TYPE, assume it's Release:
if (${CMAKE_GENERATOR} MATCHES ".*Makefiles")
    if("${CMAKE_BUILD_TYPE}" STREQUAL "")
        set(CMAKE_BUILD_TYPE Release)
    endif()
endif()

set  (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package (Qt4 REQUIRED)

set  (project_SOURCES main.cpp
                      mainwindow.cpp
                      cifar_reader.cpp
                      linearsvm.cpp
                      aux.cpp
                      linearsoftmax.cpp
                      classifier.cpp
                      simpleneuralnet.cpp
                      fisheryatesshuffle.cpp)
set  (project_HEADERS mainwindow.h
                      cifar_reader.h
                      linearsvm.h
                      CMatrix.h
                      CVector.h
                      aux.h
                      linearsoftmax.h
                      classifier.h
                      simpleneuralnet.h
                      fisheryatesshuffle.h)
set  (project_FORMS   mainwindow.ui)


QT4_WRAP_CPP (project_HEADERS_MOC   ${project_HEADERS})
QT4_WRAP_UI  (project_FORMS_HEADERS ${project_FORMS})

include_directories(${CMAKE_CURRENT_BINARY_DIR})

include         (${QT_USE_FILE})
add_definitions (${QT_DEFINITIONS})

add_executable  (${PROJECT_NAME} ${project_SOURCES}
                                 ${project_FORMS_HEADERS}
                                 ${project_HEADERS_MOC})

target_link_libraries (${PROJECT_NAME} ${QT_LIBRARIES})

# Additional targets
set(net_SOURCES simplenet_main.cpp
                simplenet_ui.cpp
                simpleneuralnet.cpp
                cifar_reader.cpp
                fisheryatesshuffle.cpp)
set(net_HEADERS simplenet_ui.h
                simpleneuralnet.h
                cifar_reader.h
                fisheryatesshuffle.h)
set(net_FORMS   simplenet_ui.ui)

QT4_WRAP_CPP (net_HEADERS_MOC   ${net_HEADERS})
QT4_WRAP_UI  (net_FORMS_HEADERS ${net_FORMS})

add_executable(simple_net ${net_SOURCES} ${net_HEADERS_MOC} ${net_FORMS_HEADERS})
target_link_libraries (simple_net ${QT_LIBRARIES})

add_executable(test_fisheryatesshuffle test_fisheryatesshuffle.cpp fisheryatesshuffle.cpp)
add_executable(test_simpleneuralnetwork test_simpleneuralnetwork.cpp simpleneuralnet.cpp)

add_executable(ensambles ensambles.cpp simpleneuralnet.cpp cifar_reader.cpp fisheryatesshuffle.cpp)
