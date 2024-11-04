cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(beginner_tutorials)
find_package(gazebo REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  gazebo_ros
  rospy
  std_msgs
  message_generation
  gazebo_msgs
)






  
include_directories(${Boost_INCLUDE_DIR} ${GAZEBO_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")




add_executable(listener listener.cc )
target_link_libraries(listener ${GAZEBO_LIBRARIES}  ${catkin_LIBRARIES} pthread)



