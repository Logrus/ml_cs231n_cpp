cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(visualizer)

# In case of Makefiles if the user does not setup CMAKE_BUILD_TYPE, assume it's Release:
if (${CMAKE_GENERATOR} MATCHES ".*Makefiles")
    if("${CMAKE_BUILD_TYPE}" STREQUAL "")
        set(CMAKE_BUILD_TYPE Release)
    endif()
endif()

set  (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")



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

add_executable(simple_net ${net_SOURCES} ${net_HEADERS_MOC} ${net_FORMS_HEADERS})
target_link_libraries (simple_net ${QT_LIBRARIES})
add_dependencies(simple_net copy_custom_plot)


add_executable(test_fisheryatesshuffle test/test_fisheryatesshuffle.cpp src/fisheryatesshuffle.cpp)
add_executable(test_simpleneuralnetwork test/test_simpleneuralnetwork.cpp src/simpleneuralnet.cpp)

add_executable(ensambles src/ensambles.cpp src/simpleneuralnet.cpp src/cifar_reader.cpp src/fisheryatesshuffle.cpp)
