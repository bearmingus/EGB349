# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/cdrone/catkin_ws/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cdrone/catkin_ws/build

# Utility rule file for _depthai_ros_msgs_generate_messages_check_deps_SpatialDetection.

# Include the progress variables for this target.
include depthai-ros/depthai_ros_msgs/CMakeFiles/_depthai_ros_msgs_generate_messages_check_deps_SpatialDetection.dir/progress.make

depthai-ros/depthai_ros_msgs/CMakeFiles/_depthai_ros_msgs_generate_messages_check_deps_SpatialDetection:
	cd /home/cdrone/catkin_ws/build/depthai-ros/depthai_ros_msgs && ../../catkin_generated/env_cached.sh /usr/bin/python3 /opt/ros/noetic/share/genmsg/cmake/../../../lib/genmsg/genmsg_check_deps.py depthai_ros_msgs /home/cdrone/catkin_ws/src/depthai-ros/depthai_ros_msgs/msg/SpatialDetection.msg geometry_msgs/Point:vision_msgs/BoundingBox2D:vision_msgs/ObjectHypothesis:geometry_msgs/Pose2D

_depthai_ros_msgs_generate_messages_check_deps_SpatialDetection: depthai-ros/depthai_ros_msgs/CMakeFiles/_depthai_ros_msgs_generate_messages_check_deps_SpatialDetection
_depthai_ros_msgs_generate_messages_check_deps_SpatialDetection: depthai-ros/depthai_ros_msgs/CMakeFiles/_depthai_ros_msgs_generate_messages_check_deps_SpatialDetection.dir/build.make

.PHONY : _depthai_ros_msgs_generate_messages_check_deps_SpatialDetection

# Rule to build all files generated by this target.
depthai-ros/depthai_ros_msgs/CMakeFiles/_depthai_ros_msgs_generate_messages_check_deps_SpatialDetection.dir/build: _depthai_ros_msgs_generate_messages_check_deps_SpatialDetection

.PHONY : depthai-ros/depthai_ros_msgs/CMakeFiles/_depthai_ros_msgs_generate_messages_check_deps_SpatialDetection.dir/build

depthai-ros/depthai_ros_msgs/CMakeFiles/_depthai_ros_msgs_generate_messages_check_deps_SpatialDetection.dir/clean:
	cd /home/cdrone/catkin_ws/build/depthai-ros/depthai_ros_msgs && $(CMAKE_COMMAND) -P CMakeFiles/_depthai_ros_msgs_generate_messages_check_deps_SpatialDetection.dir/cmake_clean.cmake
.PHONY : depthai-ros/depthai_ros_msgs/CMakeFiles/_depthai_ros_msgs_generate_messages_check_deps_SpatialDetection.dir/clean

depthai-ros/depthai_ros_msgs/CMakeFiles/_depthai_ros_msgs_generate_messages_check_deps_SpatialDetection.dir/depend:
	cd /home/cdrone/catkin_ws/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cdrone/catkin_ws/src /home/cdrone/catkin_ws/src/depthai-ros/depthai_ros_msgs /home/cdrone/catkin_ws/build /home/cdrone/catkin_ws/build/depthai-ros/depthai_ros_msgs /home/cdrone/catkin_ws/build/depthai-ros/depthai_ros_msgs/CMakeFiles/_depthai_ros_msgs_generate_messages_check_deps_SpatialDetection.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : depthai-ros/depthai_ros_msgs/CMakeFiles/_depthai_ros_msgs_generate_messages_check_deps_SpatialDetection.dir/depend

