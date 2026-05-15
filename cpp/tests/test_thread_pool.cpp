#include <cassert>
#include <numeric>
#include <vector>

#include "fasttext_threaded/thread_pool.h"

int main() {
  fasttext_threaded::ThreadPool pool(4);
  std::vector<std::future<int>> futures;

  for (int value = 0; value < 32; ++value) {
    futures.push_back(pool.submit([value]() { return value * value; }));
  }

  int sum = 0;
  for (auto& future : futures) {
    sum += future.get();
  }

  assert(sum == 10416);
  assert(pool.size() == 4);
  return 0;
}
