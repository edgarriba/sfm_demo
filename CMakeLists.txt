cmake_minimum_required(VERSION 2.8)
project( SFM_DEMO )

find_package( OpenCV REQUIRED )

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    include
    src/include/
)

# CUSTOM LIBS
add_library(feaures_lib src/robust_matcher.cpp)

# EXECUTABLES
add_executable( sfm_demo src/sfm_demo.cpp )
target_link_libraries( sfm_demo feaures_lib ${OpenCV_LIBS} )

# TESTS
add_executable( test_nn_match test/test_nn_match.cpp )
add_executable( test_robust_essential test/test_robust_essential.cpp )
add_executable( test_motion test/test_motion_estimation.cpp )

target_link_libraries( test_nn_match feaures_lib ${OpenCV_LIBS} )
target_link_libraries( test_robust_essential feaures_lib ${OpenCV_LIBS} )
target_link_libraries( test_motion feaures_lib ${OpenCV_LIBS} )
