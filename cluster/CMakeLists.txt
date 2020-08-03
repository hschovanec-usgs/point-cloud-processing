cmake_minimum_required( VERSION 2.8 )

# Create Project
project( Cluster )
add_executable( cluster cluster.cpp)
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "cluster" )

# Find Packages
find_package( PCL 1.9 REQUIRED )

if( PCL_FOUND )
  # Additional Include Directories
  # [C/C++]>[General]>[Additional Include Directories]
  include_directories( ${PCL_INCLUDE_DIRS} )

  # Preprocessor Definitions
  # [C/C++]>[Preprocessor]>[Preprocessor Definitions]
  add_definitions( ${PCL_DEFINITIONS} )
  #add_definitions( -DPCL_NO_PRECOMPILE )

  # Additional Library Directories
  # [Linker]>[General]>[Additional Library Directories]
  link_directories( ${PCL_LIBRARY_DIRS} )

  # Additional Dependencies
  # [Linker]>[Input]>[Additional Dependencies]
  target_link_libraries( cluster ${PCL_LIBRARIES} )
endif()
