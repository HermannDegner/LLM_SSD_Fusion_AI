
cmake_minimum_required(VERSION 3.12)
project(ssd_align_leap_dll LANGUAGES CXX)

add_library(ssd_align_leap SHARED ssd_align_leap_dll.cpp)
target_include_directories(ssd_align_leap PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(ssd_align_leap PRIVATE cxx_std_17)

# Windows DLL name
set_target_properties(ssd_align_leap PROPERTIES
    OUTPUT_NAME "ssd_align_leap"
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

# Example install rules
install(TARGETS ssd_align_leap
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(FILES ssd_align_leap_dll.h DESTINATION include)
