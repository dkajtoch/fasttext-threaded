add_executable(test_thread_pool
  test_thread_pool.cpp
  ../src/thread_pool.cpp
)
target_include_directories(test_thread_pool PRIVATE ../include)
if(FASTTEXT_THREADED_SANITIZER_FLAGS)
  target_compile_options(test_thread_pool PRIVATE
    ${FASTTEXT_THREADED_SANITIZER_FLAGS}
    -fno-omit-frame-pointer
  )
  target_link_options(test_thread_pool PRIVATE ${FASTTEXT_THREADED_SANITIZER_FLAGS})
endif()
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
if(FASTTEXT_THREADED_SANITIZER_FLAGS)
  target_compile_options(test_parallel_model PRIVATE
    ${FASTTEXT_THREADED_SANITIZER_FLAGS}
    -fno-omit-frame-pointer
  )
  target_link_options(test_parallel_model PRIVATE ${FASTTEXT_THREADED_SANITIZER_FLAGS})
endif()
add_test(NAME test_parallel_model COMMAND test_parallel_model)
