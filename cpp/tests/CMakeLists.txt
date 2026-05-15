add_executable(test_thread_pool
  test_thread_pool.cpp
  ../src/thread_pool.cpp
)
target_include_directories(test_thread_pool PRIVATE ../include)
add_test(NAME test_thread_pool COMMAND test_thread_pool)

add_executable(test_parallel_model
  test_parallel_model.cpp
  ../src/parallel_model.cpp
  ../src/thread_pool.cpp
)
target_include_directories(test_parallel_model PRIVATE
  ../include
)
target_link_libraries(test_parallel_model PRIVATE fasttext_vendor)
if(UNIX AND NOT APPLE)
  target_link_libraries(test_parallel_model PRIVATE pthread)
endif()
add_test(NAME test_parallel_model COMMAND test_parallel_model)
