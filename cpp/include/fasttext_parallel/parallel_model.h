#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "fasttext.h"
#include "fasttext_parallel/thread_pool.h"

namespace fasttext_parallel {

struct PredictionResult {
  std::vector<std::string> labels;
  std::vector<fasttext::real> probabilities;
};

class ParallelModel {
 public:
  ParallelModel(const std::string& model_path, std::size_t thread_count);

  ParallelModel(const ParallelModel&) = delete;
  ParallelModel& operator=(const ParallelModel&) = delete;

  std::vector<PredictionResult> predict_many(const std::vector<std::string>& lines,
                                             int32_t k, fasttext::real threshold);

  [[nodiscard]] std::size_t threads() const;

 private:
  PredictionResult predict_one(const std::string& line, int32_t k,
                               fasttext::real threshold) const;

  fasttext::FastText model_;
  ThreadPool pool_;
};

}  // namespace fasttext_parallel
