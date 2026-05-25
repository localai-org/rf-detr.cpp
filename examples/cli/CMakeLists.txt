add_executable(rfdetr-cli main.cpp)
target_link_libraries(rfdetr-cli PRIVATE rfdetr)
target_include_directories(rfdetr-cli PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src)
