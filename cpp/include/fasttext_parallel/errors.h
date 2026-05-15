#pragma once

#include <stdexcept>
#include <string>

namespace fasttext_parallel {

class FastTextParallelError : public std::runtime_error {
 public:
  explicit FastTextParallelError(const std::string& message)
      : std::runtime_error(message) {}
};

class ModelLoadError : public FastTextParallelError {
 public:
  explicit ModelLoadError(const std::string& message)
      : FastTextParallelError(message) {}
};

class PredictionError : public FastTextParallelError {
 public:
  explicit PredictionError(const std::string& message)
      : FastTextParallelError(message) {}
};

class InvalidInputError : public FastTextParallelError {
 public:
  explicit InvalidInputError(const std::string& message)
      : FastTextParallelError(message) {}
};

}  // namespace fasttext_parallel
