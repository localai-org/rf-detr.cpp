# rfdetr-video: detect/segment objects in a video file and write an annotated
# output video. ffmpeg is invoked via popen(); no libav link dependency.
add_executable(rfdetr-video main.cpp)
target_link_libraries(rfdetr-video PRIVATE rfdetr)
target_include_directories(rfdetr-video PRIVATE
    ${CMAKE_SOURCE_DIR}/include)
