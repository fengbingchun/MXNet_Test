PROJECT(MXNet_Test)
CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

# support C++11
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# support C++14, when gcc version > 5.1, use -std=c++14 instead of c++1y
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y")

IF(NOT CMAKE_BUILD_TYPE)
	SET(CMAKE_BUILD_TYPE "Release")
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O2")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")
ELSE()
	SET(CMAKE_BUILD_TYPE "Debug")
	SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -Wall -O2")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -O2")
ENDIF()
MESSAGE(STATUS "cmake build type: ${CMAKE_BUILD_TYPE}")

MESSAGE(STATUS "cmake current source dir: ${CMAKE_CURRENT_SOURCE_DIR}")
SET(PATH_TEST_FILES ${CMAKE_CURRENT_SOURCE_DIR}/./../../demo/MXNet_Test)
SET(PATH_SRC_DMLC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/./../../src/dmlc-core)
SET(PATH_SRC_MSHADOW_FILES ${CMAKE_CURRENT_SOURCE_DIR}/./../../src/mshadow)
SET(PATH_SRC_OPENBLAS_FILES ${CMAKE_CURRENT_SOURCE_DIR}/./../../src/openblas)
SET(PATH_SRC_MXNET_FILES ${CMAKE_CURRENT_SOURCE_DIR}/./../../src/mxnet)
SET(PATH_SRC_TVM_FILES ${CMAKE_CURRENT_SOURCE_DIR}/./../../src/tvm)
SET(PATH_SRC_DLPACK_FILES ${CMAKE_CURRENT_SOURCE_DIR}/./../../src/dlpack)
MESSAGE(STATUS "path test files: ${PATH_TEST_FILES}")

# don't use opencv in mxnet
ADD_DEFINITIONS(-DMXNET_USE_OPENCV=0)
ADD_DEFINITIONS(-DMSHADOW_USE_F16C=0)

SET(PATH_OPENCV /opt/opencv3.4.2)
IF(EXISTS ${PATH_OPENCV})
	MESSAGE(STATUS "Found OpenCV: ${PATH_OPENCV}")
ELSE()
	MESSAGE(FATAL_ERROR "Can not find OpenCV in ${PATH_OPENCV}")
ENDIF()

# head file search path
INCLUDE_DIRECTORIES(
	${PATH_SRC_OPENBLAS_FILES}
	${PATH_SRC_OPENBLAS_FILES}/build # include openblas config.h
	${PATH_SRC_DLPACK_FILES}/include
	${PATH_SRC_MSHADOW_FILES}
	${PATH_SRC_DMLC_FILES}/include
	${PATH_SRC_TVM_FILES}/include
	${PATH_SRC_TVM_FILES}/nnvm/include
	${PATH_SRC_MXNET_FILES}/src
	${PATH_SRC_MXNET_FILES}/include
	${PATH_SRC_MXNET_FILES}/cpp-package/include
	${PATH_OPENCV}/include
	${PATH_TEST_FILES}
)

# build mxnet dynamic library
SET(MXNET_SRC_LIST )

# tvm
FILE(GLOB_RECURSE SRC_TVM_NNVM_C_API ${PATH_SRC_TVM_FILES}/nnvm/src/c_api/*.cc)
FILE(GLOB_RECURSE SRC_TVM_NNVM_CORE ${PATH_SRC_TVM_FILES}/nnvm/src/core/*.cc)
FILE(GLOB_RECURSE SRC_TVM_NNVM_PASS ${PATH_SRC_TVM_FILES}/nnvm/src/pass/*.cc)

# mxnet
FILE(GLOB_RECURSE SRC_MXNET ${PATH_SRC_MXNET_FILES}/src/*.cc)

# dmlc
FILE(GLOB SRC_DMLC ${PATH_SRC_DMLC_FILES}/src/*.cc)
FILE(GLOB_RECURSE SRC_DMLC_INCLUDE ${PATH_SRC_DMLC_FILES}/include/*)
LIST(APPEND SRC_DMLC ${SRC_DMLC_INCLUDE})
LIST(APPEND SRC_DMLC ${PATH_SRC_DMLC_FILES}/src/io/line_split.cc)
LIST(APPEND SRC_DMLC ${PATH_SRC_DMLC_FILES}/src/io/recordio_split.cc)
LIST(APPEND SRC_DMLC ${PATH_SRC_DMLC_FILES}/src/io/indexed_recordio_split.cc)
LIST(APPEND SRC_DMLC ${PATH_SRC_DMLC_FILES}/src/io/input_split_base.cc)
LIST(APPEND SRC_DMLC ${PATH_SRC_DMLC_FILES}/src/io/filesys.cc)
LIST(APPEND SRC_DMLC ${PATH_SRC_DMLC_FILES}/src/io/local_filesys.cc)

LIST(APPEND MXNET_SRC_LIST
	${SRC_TVM_NNVM_C_API}
	${SRC_TVM_NNVM_CORE}
	${SRC_TVM_NNVM_PASS}
	${SRC_MXNET}
	${SRC_DMLC}
)
#MESSAGE(STATUS "###### src: ${MXNET_SRC_LIST}")

ADD_LIBRARY(mxnet SHARED ${MXNET_SRC_LIST})

# find opencv library
FIND_LIBRARY(opencv_core NAMES opencv_core PATHS ${PATH_OPENCV}/lib NO_DEFAULT_PATH)
FIND_LIBRARY(opencv_imgproc NAMES opencv_imgproc PATHS ${PATH_OPENCV}/lib NO_DEFAULT_PATH)
FIND_LIBRARY(opencv_highgui NAMES opencv_highgui PATHS ${PATH_OPENCV}/lib NO_DEFAULT_PATH)
FIND_LIBRARY(opencv_imgcodecs NAMES opencv_imgcodecs PATHS ${PATH_OPENCV}/lib NO_DEFAULT_PATH)
FIND_LIBRARY(opencv_video NAMES opencv_video PATHS ${PATH_OPENCV}/lib NO_DEFAULT_PATH)
FIND_LIBRARY(opencv_videoio NAMES opencv_videoio PATHS ${PATH_OPENCV}/lib NO_DEFAULT_PATH)
FIND_LIBRARY(opencv_objdetect NAMES opencv_objdetect PATHS ${PATH_OPENCV}/lib NO_DEFAULT_PATH)
FIND_LIBRARY(opencv_ml NAMES opencv_ml PATHS ${PATH_OPENCV}/lib NO_DEFAULT_PATH)
MESSAGE(STATUS "opencv libraries: ${opencv_core} ${opencv_imgproc} ${opencv_highgui} ${opencv_imgcodecs} ${opencv_video} ${opencv_videoio} ${opencv_objdetect} ${opencv_ml}")

# find dep library
SET(DEP_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/build CACHE PATH "dep library path")
MESSAGE(STATUS "dep library dir: ${DEP_LIB_DIR}")
LINK_DIRECTORIES(${DEP_LIB_DIR})

# recursive query match files :*.cpp, *.cc
FILE(GLOB_RECURSE TEST_CPP_LIST ${PATH_TEST_FILES}/*.cpp)
FILE(GLOB_RECURSE TEST_CC_LIST ${PATH_TEST_FILES}/*.cc)
MESSAGE(STATUS "test cpp list: ${TEST_CPP_LIST} ${TEST_C_LIST}")

# build executable program
ADD_EXECUTABLE(MXNet_Test ${TEST_CPP_LIST} ${TEST_CC_LIST})
# add dependent library: static and dynamic
TARGET_LINK_LIBRARIES(MXNet_Test
	mxnet
	${DEP_LIB_DIR}/libopenblas.so
	pthread
	rt # undefined reference to shm_open, need to be placed in front of opencv library
	${opencv_core}
	${opencv_imgproc}
	${opencv_highgui}
	${opencv_imgcodecs}
	${opencv_video}
	${opencv_videoio}
	${opencv_objdetect}
	${opencv_ml}
)

