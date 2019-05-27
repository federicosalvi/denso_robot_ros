cmake_minimum_required(VERSION 2.8.3)
project(pose_dataset_tool)

find_package(catkin REQUIRED COMPONENTS
  rospy
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES pose_dataset_tool
#  CATKIN_DEPENDS rospy
#  DEPENDS system_lib
)

## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

# launch
install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

# script
file(GLOB prgs RELATIVE ${PROJECT_SOURCE_DIR} "scripts/*.py")
foreach(prg ${prgs})
   catkin_install_python(PROGRAMS ${prg}
      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
endforeach()