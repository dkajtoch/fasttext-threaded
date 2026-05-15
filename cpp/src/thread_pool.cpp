#include "fasttext_parallel/thread_pool.h"

#include <stdexcept>

namespace fasttext_parallel {

ThreadPool::ThreadPool(std::size_t thread_count) {
  if (thread_count == 0) {
    throw std::invalid_argument("thread_count must be greater than zero");
  }

  workers_.reserve(thread_count);
  for (std::size_t index = 0; index < thread_count; ++index) {
    workers_.emplace_back([this]() { worker_loop(); });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }

  condition_.notify_all();
  for (std::thread& worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

std::size_t ThreadPool::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return workers_.size();
}

void ThreadPool::worker_loop() {
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      condition_.wait(lock, [this]() { return stop_ || !tasks_.empty(); });

      if (stop_ && tasks_.empty()) {
        return;
      }

      task = std::move(tasks_.front());
      tasks_.pop_front();
    }

    task();
  }
}

}  // namespace fasttext_parallel
