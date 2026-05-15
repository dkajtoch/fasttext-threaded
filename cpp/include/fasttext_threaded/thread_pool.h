#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

namespace fasttext_threaded {

class ThreadPool {
 public:
  explicit ThreadPool(std::size_t thread_count);
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  template <class Function>
  auto submit(Function&& function) -> std::future<std::invoke_result_t<Function>> {
    using Result = std::invoke_result_t<Function>;

    auto task = std::make_shared<std::packaged_task<Result()>>(
        std::forward<Function>(function));
    std::future<Result> future = task->get_future();

    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stop_) {
        throw std::runtime_error("cannot submit work to a stopped thread pool");
      }
      tasks_.emplace_back([task]() { (*task)(); });
    }

    condition_.notify_one();
    return future;
  }

  [[nodiscard]] std::size_t size() const;

 private:
  void worker_loop();

  std::vector<std::thread> workers_;
  std::deque<std::function<void()>> tasks_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  bool stop_{false};
};

}  // namespace fasttext_threaded
