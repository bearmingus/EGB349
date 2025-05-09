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

# Include any dependencies generated for this target.
include depthai-ros/depthai_examples/CMakeFiles/rgb_node.dir/depend.make

# Include the progress variables for this target.
include depthai-ros/depthai_examples/CMakeFiles/rgb_node.dir/progress.make

# Include the compile flags for this target's objects.
include depthai-ros/depthai_examples/CMakeFiles/rgb_node.dir/flags.make

depthai-ros/depthai_examples/CMakeFiles/rgb_node.dir/src/rgb_publisher.cpp.o: depthai-ros/depthai_examples/CMakeFiles/rgb_node.dir/flags.make
depthai-ros/depthai_examples/CMakeFiles/rgb_node.dir/src/rgb_publisher.cpp.o: /home/cdrone/catkin_ws/src/depthai-ros/depthai_examples/src/rgb_publisher.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cdrone/catkin_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object depthai-ros/depthai_examples/CMakeFiles/rgb_node.dir/src/rgb_publisher.cpp.o"
	cd /home/cdrone/catkin_ws/build/depthai-ros/depthai_examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/rgb_node.dir/src/rgb_publisher.cpp.o -c /home/cdrone/catkin_ws/src/depthai-ros/depthai_examples/src/rgb_publisher.cpp

depthai-ros/depthai_examples/CMakeFiles/rgb_node.dir/src/rgb_publisher.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rgb_node.dir/src/rgb_publisher.cpp.i"
	cd /home/cdrone/catkin_ws/build/depthai-ros/depthai_examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cdrone/catkin_ws/src/depthai-ros/depthai_examples/src/rgb_publisher.cpp > CMakeFiles/rgb_node.dir/src/rgb_publisher.cpp.i

depthai-ros/depthai_examples/CMakeFiles/rgb_node.dir/src/rgb_publisher.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rgb_node.dir/src/rgb_publisher.cpp.s"
	cd /home/cdrone/catkin_ws/build/depthai-ros/depthai_examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cdrone/catkin_ws/src/depthai-ros/depthai_examples/src/rgb_publisher.cpp -o CMakeFiles/rgb_node.dir/src/rgb_publisher.cpp.s

# Object files for target rgb_node
rgb_node_OBJECTS = \
"CMakeFiles/rgb_node.dir/src/rgb_publisher.cpp.o"

# External object files for target rgb_node
rgb_node_EXTERNAL_OBJECTS =

/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: depthai-ros/depthai_examples/CMakeFiles/rgb_node.dir/src/rgb_publisher.cpp.o
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: depthai-ros/depthai_examples/CMakeFiles/rgb_node.dir/build.make
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /home/cdrone/catkin_ws/devel/lib/libdepthai_bridge.so
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /opt/ros/noetic/lib/libcamera_info_manager.so
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /opt/ros/noetic/lib/libcamera_calibration_parsers.so
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /opt/ros/noetic/lib/libimage_transport.so
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /opt/ros/noetic/lib/libcv_bridge.so
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_calib3d.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_dnn.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_features2d.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_flann.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_highgui.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_ml.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_objdetect.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_photo.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_stitching.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_video.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_aruco.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_bgsegm.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_bioinspired.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_ccalib.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_datasets.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_dnn_objdetect.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_dnn_superres.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_dpm.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_face.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_freetype.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_fuzzy.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_hdf.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_hfs.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_img_hash.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_line_descriptor.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_optflow.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_phase_unwrapping.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_plot.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_quality.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_reg.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_rgbd.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_saliency.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_shape.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_stereo.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_structured_light.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_superres.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_surface_matching.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_text.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_tracking.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_videostab.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_viz.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_ximgproc.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_xobjdetect.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_xphoto.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /opt/ros/noetic/lib/libmessage_filters.so
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /opt/ros/noetic/lib/libnodeletlib.so
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /opt/ros/noetic/lib/libbondcpp.so
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libuuid.so
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /opt/ros/noetic/lib/libclass_loader.so
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libPocoFoundation.so
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libdl.so
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /opt/ros/noetic/lib/libroslib.so
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /opt/ros/noetic/lib/librospack.so
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libpython3.8.so
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libboost_program_options.so.1.71.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libtinyxml2.so
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /opt/ros/noetic/lib/libroscpp.so
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libpthread.so
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libboost_chrono.so.1.71.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libboost_filesystem.so.1.71.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /opt/ros/noetic/lib/librosconsole.so
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /opt/ros/noetic/lib/librosconsole_log4cxx.so
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /opt/ros/noetic/lib/librosconsole_backend_interface.so
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/liblog4cxx.so
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libboost_regex.so.1.71.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /opt/ros/noetic/lib/libxmlrpcpp.so
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /opt/ros/noetic/lib/libroscpp_serialization.so
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /opt/ros/noetic/lib/librostime.so
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libboost_date_time.so.1.71.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /opt/ros/noetic/lib/libcpp_common.so
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libboost_system.so.1.71.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libboost_thread.so.1.71.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libconsole_bridge.so.0.4
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /opt/ros/noetic/lib/aarch64-linux-gnu/libdepthai-core.so
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_highgui.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /opt/ros/noetic/lib/aarch64-linux-gnu/libdepthai-core.so
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_highgui.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.2.0
/home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node: depthai-ros/depthai_examples/CMakeFiles/rgb_node.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cdrone/catkin_ws/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable /home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node"
	cd /home/cdrone/catkin_ws/build/depthai-ros/depthai_examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/rgb_node.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
depthai-ros/depthai_examples/CMakeFiles/rgb_node.dir/build: /home/cdrone/catkin_ws/devel/lib/depthai_examples/rgb_node

.PHONY : depthai-ros/depthai_examples/CMakeFiles/rgb_node.dir/build

depthai-ros/depthai_examples/CMakeFiles/rgb_node.dir/clean:
	cd /home/cdrone/catkin_ws/build/depthai-ros/depthai_examples && $(CMAKE_COMMAND) -P CMakeFiles/rgb_node.dir/cmake_clean.cmake
.PHONY : depthai-ros/depthai_examples/CMakeFiles/rgb_node.dir/clean

depthai-ros/depthai_examples/CMakeFiles/rgb_node.dir/depend:
	cd /home/cdrone/catkin_ws/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cdrone/catkin_ws/src /home/cdrone/catkin_ws/src/depthai-ros/depthai_examples /home/cdrone/catkin_ws/build /home/cdrone/catkin_ws/build/depthai-ros/depthai_examples /home/cdrone/catkin_ws/build/depthai-ros/depthai_examples/CMakeFiles/rgb_node.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : depthai-ros/depthai_examples/CMakeFiles/rgb_node.dir/depend

