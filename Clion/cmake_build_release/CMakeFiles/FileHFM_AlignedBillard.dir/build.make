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
CMAKE_COMMAND = /home/liyx0l/Applications/clion/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/liyx0l/Applications/clion/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/liyx0l/Applications/hfm/Interfaces/FileHFM

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/liyx0l/Applications/hfm/Clion/cmake_build_release

# Include any dependencies generated for this target.
include CMakeFiles/FileHFM_AlignedBillard.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/FileHFM_AlignedBillard.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/FileHFM_AlignedBillard.dir/flags.make

CMakeFiles/FileHFM_AlignedBillard.dir/FileHFM.cpp.o: CMakeFiles/FileHFM_AlignedBillard.dir/flags.make
CMakeFiles/FileHFM_AlignedBillard.dir/FileHFM.cpp.o: /home/liyx0l/Applications/hfm/Interfaces/FileHFM/FileHFM.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liyx0l/Applications/hfm/Clion/cmake_build_release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/FileHFM_AlignedBillard.dir/FileHFM.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/FileHFM_AlignedBillard.dir/FileHFM.cpp.o -c /home/liyx0l/Applications/hfm/Interfaces/FileHFM/FileHFM.cpp

CMakeFiles/FileHFM_AlignedBillard.dir/FileHFM.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FileHFM_AlignedBillard.dir/FileHFM.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liyx0l/Applications/hfm/Interfaces/FileHFM/FileHFM.cpp > CMakeFiles/FileHFM_AlignedBillard.dir/FileHFM.cpp.i

CMakeFiles/FileHFM_AlignedBillard.dir/FileHFM.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FileHFM_AlignedBillard.dir/FileHFM.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liyx0l/Applications/hfm/Interfaces/FileHFM/FileHFM.cpp -o CMakeFiles/FileHFM_AlignedBillard.dir/FileHFM.cpp.s

# Object files for target FileHFM_AlignedBillard
FileHFM_AlignedBillard_OBJECTS = \
"CMakeFiles/FileHFM_AlignedBillard.dir/FileHFM.cpp.o"

# External object files for target FileHFM_AlignedBillard
FileHFM_AlignedBillard_EXTERNAL_OBJECTS =

FileHFM_AlignedBillard: CMakeFiles/FileHFM_AlignedBillard.dir/FileHFM.cpp.o
FileHFM_AlignedBillard: CMakeFiles/FileHFM_AlignedBillard.dir/build.make
FileHFM_AlignedBillard: CMakeFiles/FileHFM_AlignedBillard.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/liyx0l/Applications/hfm/Clion/cmake_build_release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable FileHFM_AlignedBillard"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/FileHFM_AlignedBillard.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/FileHFM_AlignedBillard.dir/build: FileHFM_AlignedBillard

.PHONY : CMakeFiles/FileHFM_AlignedBillard.dir/build

CMakeFiles/FileHFM_AlignedBillard.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/FileHFM_AlignedBillard.dir/cmake_clean.cmake
.PHONY : CMakeFiles/FileHFM_AlignedBillard.dir/clean

CMakeFiles/FileHFM_AlignedBillard.dir/depend:
	cd /home/liyx0l/Applications/hfm/Clion/cmake_build_release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/liyx0l/Applications/hfm/Interfaces/FileHFM /home/liyx0l/Applications/hfm/Interfaces/FileHFM /home/liyx0l/Applications/hfm/Clion/cmake_build_release /home/liyx0l/Applications/hfm/Clion/cmake_build_release /home/liyx0l/Applications/hfm/Clion/cmake_build_release/CMakeFiles/FileHFM_AlignedBillard.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/FileHFM_AlignedBillard.dir/depend
