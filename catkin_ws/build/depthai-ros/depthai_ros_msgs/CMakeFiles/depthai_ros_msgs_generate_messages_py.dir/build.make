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

# Utility rule file for depthai_ros_msgs_generate_messages_py.

# Include the progress variables for this target.
include depthai-ros/depthai_ros_msgs/CMakeFiles/depthai_ros_msgs_generate_messages_py.dir/progress.make

depthai-ros/depthai_ros_msgs/CMakeFiles/depthai_ros_msgs_generate_messages_py: /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_AutoFocusCtrl.py
depthai-ros/depthai_ros_msgs/CMakeFiles/depthai_ros_msgs_generate_messages_py: /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_HandLandmark.py
depthai-ros/depthai_ros_msgs/CMakeFiles/depthai_ros_msgs_generate_messages_py: /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_HandLandmarkArray.py
depthai-ros/depthai_ros_msgs/CMakeFiles/depthai_ros_msgs_generate_messages_py: /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_SpatialDetection.py
depthai-ros/depthai_ros_msgs/CMakeFiles/depthai_ros_msgs_generate_messages_py: /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_SpatialDetectionArray.py
depthai-ros/depthai_ros_msgs/CMakeFiles/depthai_ros_msgs_generate_messages_py: /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/srv/_TriggerNamed.py
depthai-ros/depthai_ros_msgs/CMakeFiles/depthai_ros_msgs_generate_messages_py: /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/srv/_NormalizedImageCrop.py
depthai-ros/depthai_ros_msgs/CMakeFiles/depthai_ros_msgs_generate_messages_py: /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/__init__.py
depthai-ros/depthai_ros_msgs/CMakeFiles/depthai_ros_msgs_generate_messages_py: /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/srv/__init__.py


/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_AutoFocusCtrl.py: /opt/ros/noetic/lib/genpy/genmsg_py.py
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_AutoFocusCtrl.py: /home/cdrone/catkin_ws/src/depthai-ros/depthai_ros_msgs/msg/AutoFocusCtrl.msg
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/cdrone/catkin_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating Python from MSG depthai_ros_msgs/AutoFocusCtrl"
	cd /home/cdrone/catkin_ws/build/depthai-ros/depthai_ros_msgs && ../../catkin_generated/env_cached.sh /usr/bin/python3 /opt/ros/noetic/share/genpy/cmake/../../../lib/genpy/genmsg_py.py /home/cdrone/catkin_ws/src/depthai-ros/depthai_ros_msgs/msg/AutoFocusCtrl.msg -Idepthai_ros_msgs:/home/cdrone/catkin_ws/src/depthai-ros/depthai_ros_msgs/msg -Igeometry_msgs:/opt/ros/noetic/share/geometry_msgs/cmake/../msg -Isensor_msgs:/opt/ros/noetic/share/sensor_msgs/cmake/../msg -Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg -Ivision_msgs:/opt/ros/noetic/share/vision_msgs/cmake/../msg -p depthai_ros_msgs -o /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg

/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_HandLandmark.py: /opt/ros/noetic/lib/genpy/genmsg_py.py
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_HandLandmark.py: /home/cdrone/catkin_ws/src/depthai-ros/depthai_ros_msgs/msg/HandLandmark.msg
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_HandLandmark.py: /opt/ros/noetic/share/geometry_msgs/msg/Point.msg
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_HandLandmark.py: /opt/ros/noetic/share/geometry_msgs/msg/Pose2D.msg
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/cdrone/catkin_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Generating Python from MSG depthai_ros_msgs/HandLandmark"
	cd /home/cdrone/catkin_ws/build/depthai-ros/depthai_ros_msgs && ../../catkin_generated/env_cached.sh /usr/bin/python3 /opt/ros/noetic/share/genpy/cmake/../../../lib/genpy/genmsg_py.py /home/cdrone/catkin_ws/src/depthai-ros/depthai_ros_msgs/msg/HandLandmark.msg -Idepthai_ros_msgs:/home/cdrone/catkin_ws/src/depthai-ros/depthai_ros_msgs/msg -Igeometry_msgs:/opt/ros/noetic/share/geometry_msgs/cmake/../msg -Isensor_msgs:/opt/ros/noetic/share/sensor_msgs/cmake/../msg -Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg -Ivision_msgs:/opt/ros/noetic/share/vision_msgs/cmake/../msg -p depthai_ros_msgs -o /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg

/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_HandLandmarkArray.py: /opt/ros/noetic/lib/genpy/genmsg_py.py
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_HandLandmarkArray.py: /home/cdrone/catkin_ws/src/depthai-ros/depthai_ros_msgs/msg/HandLandmarkArray.msg
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_HandLandmarkArray.py: /home/cdrone/catkin_ws/src/depthai-ros/depthai_ros_msgs/msg/HandLandmark.msg
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_HandLandmarkArray.py: /opt/ros/noetic/share/geometry_msgs/msg/Point.msg
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_HandLandmarkArray.py: /opt/ros/noetic/share/geometry_msgs/msg/Pose2D.msg
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_HandLandmarkArray.py: /opt/ros/noetic/share/std_msgs/msg/Header.msg
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/cdrone/catkin_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Generating Python from MSG depthai_ros_msgs/HandLandmarkArray"
	cd /home/cdrone/catkin_ws/build/depthai-ros/depthai_ros_msgs && ../../catkin_generated/env_cached.sh /usr/bin/python3 /opt/ros/noetic/share/genpy/cmake/../../../lib/genpy/genmsg_py.py /home/cdrone/catkin_ws/src/depthai-ros/depthai_ros_msgs/msg/HandLandmarkArray.msg -Idepthai_ros_msgs:/home/cdrone/catkin_ws/src/depthai-ros/depthai_ros_msgs/msg -Igeometry_msgs:/opt/ros/noetic/share/geometry_msgs/cmake/../msg -Isensor_msgs:/opt/ros/noetic/share/sensor_msgs/cmake/../msg -Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg -Ivision_msgs:/opt/ros/noetic/share/vision_msgs/cmake/../msg -p depthai_ros_msgs -o /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg

/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_SpatialDetection.py: /opt/ros/noetic/lib/genpy/genmsg_py.py
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_SpatialDetection.py: /home/cdrone/catkin_ws/src/depthai-ros/depthai_ros_msgs/msg/SpatialDetection.msg
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_SpatialDetection.py: /opt/ros/noetic/share/geometry_msgs/msg/Point.msg
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_SpatialDetection.py: /opt/ros/noetic/share/vision_msgs/msg/BoundingBox2D.msg
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_SpatialDetection.py: /opt/ros/noetic/share/vision_msgs/msg/ObjectHypothesis.msg
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_SpatialDetection.py: /opt/ros/noetic/share/geometry_msgs/msg/Pose2D.msg
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/cdrone/catkin_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Generating Python from MSG depthai_ros_msgs/SpatialDetection"
	cd /home/cdrone/catkin_ws/build/depthai-ros/depthai_ros_msgs && ../../catkin_generated/env_cached.sh /usr/bin/python3 /opt/ros/noetic/share/genpy/cmake/../../../lib/genpy/genmsg_py.py /home/cdrone/catkin_ws/src/depthai-ros/depthai_ros_msgs/msg/SpatialDetection.msg -Idepthai_ros_msgs:/home/cdrone/catkin_ws/src/depthai-ros/depthai_ros_msgs/msg -Igeometry_msgs:/opt/ros/noetic/share/geometry_msgs/cmake/../msg -Isensor_msgs:/opt/ros/noetic/share/sensor_msgs/cmake/../msg -Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg -Ivision_msgs:/opt/ros/noetic/share/vision_msgs/cmake/../msg -p depthai_ros_msgs -o /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg

/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_SpatialDetectionArray.py: /opt/ros/noetic/lib/genpy/genmsg_py.py
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_SpatialDetectionArray.py: /home/cdrone/catkin_ws/src/depthai-ros/depthai_ros_msgs/msg/SpatialDetectionArray.msg
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_SpatialDetectionArray.py: /home/cdrone/catkin_ws/src/depthai-ros/depthai_ros_msgs/msg/SpatialDetection.msg
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_SpatialDetectionArray.py: /opt/ros/noetic/share/geometry_msgs/msg/Pose2D.msg
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_SpatialDetectionArray.py: /opt/ros/noetic/share/vision_msgs/msg/ObjectHypothesis.msg
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_SpatialDetectionArray.py: /opt/ros/noetic/share/geometry_msgs/msg/Point.msg
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_SpatialDetectionArray.py: /opt/ros/noetic/share/vision_msgs/msg/BoundingBox2D.msg
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_SpatialDetectionArray.py: /opt/ros/noetic/share/std_msgs/msg/Header.msg
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/cdrone/catkin_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Generating Python from MSG depthai_ros_msgs/SpatialDetectionArray"
	cd /home/cdrone/catkin_ws/build/depthai-ros/depthai_ros_msgs && ../../catkin_generated/env_cached.sh /usr/bin/python3 /opt/ros/noetic/share/genpy/cmake/../../../lib/genpy/genmsg_py.py /home/cdrone/catkin_ws/src/depthai-ros/depthai_ros_msgs/msg/SpatialDetectionArray.msg -Idepthai_ros_msgs:/home/cdrone/catkin_ws/src/depthai-ros/depthai_ros_msgs/msg -Igeometry_msgs:/opt/ros/noetic/share/geometry_msgs/cmake/../msg -Isensor_msgs:/opt/ros/noetic/share/sensor_msgs/cmake/../msg -Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg -Ivision_msgs:/opt/ros/noetic/share/vision_msgs/cmake/../msg -p depthai_ros_msgs -o /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg

/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/srv/_TriggerNamed.py: /opt/ros/noetic/lib/genpy/gensrv_py.py
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/srv/_TriggerNamed.py: /home/cdrone/catkin_ws/src/depthai-ros/depthai_ros_msgs/srv/TriggerNamed.srv
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/cdrone/catkin_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Generating Python code from SRV depthai_ros_msgs/TriggerNamed"
	cd /home/cdrone/catkin_ws/build/depthai-ros/depthai_ros_msgs && ../../catkin_generated/env_cached.sh /usr/bin/python3 /opt/ros/noetic/share/genpy/cmake/../../../lib/genpy/gensrv_py.py /home/cdrone/catkin_ws/src/depthai-ros/depthai_ros_msgs/srv/TriggerNamed.srv -Idepthai_ros_msgs:/home/cdrone/catkin_ws/src/depthai-ros/depthai_ros_msgs/msg -Igeometry_msgs:/opt/ros/noetic/share/geometry_msgs/cmake/../msg -Isensor_msgs:/opt/ros/noetic/share/sensor_msgs/cmake/../msg -Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg -Ivision_msgs:/opt/ros/noetic/share/vision_msgs/cmake/../msg -p depthai_ros_msgs -o /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/srv

/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/srv/_NormalizedImageCrop.py: /opt/ros/noetic/lib/genpy/gensrv_py.py
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/srv/_NormalizedImageCrop.py: /home/cdrone/catkin_ws/src/depthai-ros/depthai_ros_msgs/srv/NormalizedImageCrop.srv
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/srv/_NormalizedImageCrop.py: /opt/ros/noetic/share/geometry_msgs/msg/Pose2D.msg
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/cdrone/catkin_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Generating Python code from SRV depthai_ros_msgs/NormalizedImageCrop"
	cd /home/cdrone/catkin_ws/build/depthai-ros/depthai_ros_msgs && ../../catkin_generated/env_cached.sh /usr/bin/python3 /opt/ros/noetic/share/genpy/cmake/../../../lib/genpy/gensrv_py.py /home/cdrone/catkin_ws/src/depthai-ros/depthai_ros_msgs/srv/NormalizedImageCrop.srv -Idepthai_ros_msgs:/home/cdrone/catkin_ws/src/depthai-ros/depthai_ros_msgs/msg -Igeometry_msgs:/opt/ros/noetic/share/geometry_msgs/cmake/../msg -Isensor_msgs:/opt/ros/noetic/share/sensor_msgs/cmake/../msg -Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg -Ivision_msgs:/opt/ros/noetic/share/vision_msgs/cmake/../msg -p depthai_ros_msgs -o /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/srv

/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/__init__.py: /opt/ros/noetic/lib/genpy/genmsg_py.py
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/__init__.py: /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_AutoFocusCtrl.py
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/__init__.py: /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_HandLandmark.py
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/__init__.py: /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_HandLandmarkArray.py
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/__init__.py: /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_SpatialDetection.py
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/__init__.py: /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_SpatialDetectionArray.py
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/__init__.py: /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/srv/_TriggerNamed.py
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/__init__.py: /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/srv/_NormalizedImageCrop.py
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/cdrone/catkin_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Generating Python msg __init__.py for depthai_ros_msgs"
	cd /home/cdrone/catkin_ws/build/depthai-ros/depthai_ros_msgs && ../../catkin_generated/env_cached.sh /usr/bin/python3 /opt/ros/noetic/share/genpy/cmake/../../../lib/genpy/genmsg_py.py -o /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg --initpy

/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/srv/__init__.py: /opt/ros/noetic/lib/genpy/genmsg_py.py
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/srv/__init__.py: /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_AutoFocusCtrl.py
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/srv/__init__.py: /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_HandLandmark.py
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/srv/__init__.py: /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_HandLandmarkArray.py
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/srv/__init__.py: /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_SpatialDetection.py
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/srv/__init__.py: /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_SpatialDetectionArray.py
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/srv/__init__.py: /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/srv/_TriggerNamed.py
/home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/srv/__init__.py: /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/srv/_NormalizedImageCrop.py
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/cdrone/catkin_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Generating Python srv __init__.py for depthai_ros_msgs"
	cd /home/cdrone/catkin_ws/build/depthai-ros/depthai_ros_msgs && ../../catkin_generated/env_cached.sh /usr/bin/python3 /opt/ros/noetic/share/genpy/cmake/../../../lib/genpy/genmsg_py.py -o /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/srv --initpy

depthai_ros_msgs_generate_messages_py: depthai-ros/depthai_ros_msgs/CMakeFiles/depthai_ros_msgs_generate_messages_py
depthai_ros_msgs_generate_messages_py: /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_AutoFocusCtrl.py
depthai_ros_msgs_generate_messages_py: /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_HandLandmark.py
depthai_ros_msgs_generate_messages_py: /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_HandLandmarkArray.py
depthai_ros_msgs_generate_messages_py: /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_SpatialDetection.py
depthai_ros_msgs_generate_messages_py: /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/_SpatialDetectionArray.py
depthai_ros_msgs_generate_messages_py: /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/srv/_TriggerNamed.py
depthai_ros_msgs_generate_messages_py: /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/srv/_NormalizedImageCrop.py
depthai_ros_msgs_generate_messages_py: /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/msg/__init__.py
depthai_ros_msgs_generate_messages_py: /home/cdrone/catkin_ws/devel/lib/python3/dist-packages/depthai_ros_msgs/srv/__init__.py
depthai_ros_msgs_generate_messages_py: depthai-ros/depthai_ros_msgs/CMakeFiles/depthai_ros_msgs_generate_messages_py.dir/build.make

.PHONY : depthai_ros_msgs_generate_messages_py

# Rule to build all files generated by this target.
depthai-ros/depthai_ros_msgs/CMakeFiles/depthai_ros_msgs_generate_messages_py.dir/build: depthai_ros_msgs_generate_messages_py

.PHONY : depthai-ros/depthai_ros_msgs/CMakeFiles/depthai_ros_msgs_generate_messages_py.dir/build

depthai-ros/depthai_ros_msgs/CMakeFiles/depthai_ros_msgs_generate_messages_py.dir/clean:
	cd /home/cdrone/catkin_ws/build/depthai-ros/depthai_ros_msgs && $(CMAKE_COMMAND) -P CMakeFiles/depthai_ros_msgs_generate_messages_py.dir/cmake_clean.cmake
.PHONY : depthai-ros/depthai_ros_msgs/CMakeFiles/depthai_ros_msgs_generate_messages_py.dir/clean

depthai-ros/depthai_ros_msgs/CMakeFiles/depthai_ros_msgs_generate_messages_py.dir/depend:
	cd /home/cdrone/catkin_ws/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cdrone/catkin_ws/src /home/cdrone/catkin_ws/src/depthai-ros/depthai_ros_msgs /home/cdrone/catkin_ws/build /home/cdrone/catkin_ws/build/depthai-ros/depthai_ros_msgs /home/cdrone/catkin_ws/build/depthai-ros/depthai_ros_msgs/CMakeFiles/depthai_ros_msgs_generate_messages_py.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : depthai-ros/depthai_ros_msgs/CMakeFiles/depthai_ros_msgs_generate_messages_py.dir/depend

