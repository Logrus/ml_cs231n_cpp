cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(visualizer)

# In case of Makefiles if the user does not setup CMAKE_BUILD_TYPE, assume it's Release:
#if (${CMAKE_GENERATOR} MATCHES ".*Makefiles")
#    if("${CMAKE_BUILD_TYPE}" STREQUAL "")
#        set(CMAKE_BUILD_TYPE Release)
#    endif()
#endif()

set  (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package (Qt4 REQUIRED)

set  (project_SOURCES main.cpp mainwindow.cpp cifar_reader.cpp linearsvm.cpp)
set  (project_HEADERS mainwindow.h cifar_reader.h linearsvm.h CMatrix.h CVector.h)
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
