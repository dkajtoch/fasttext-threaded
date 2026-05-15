#pragma once

#include <stdexcept>
#include <string>

namespace fasttext_threaded {

class FastTextThreadedError : public std::runtime_error {
 public:
  explicit FastTextThreadedError(const std::string& message)
      : std::runtime_error(message) {}
};

class ModelLoadError : public FastTextThreadedError {
 public:
  explicit ModelLoadError(const std::string& message)
      : FastTextThreadedError(message) {}
};

class PredictionError : public FastTextThreadedError {
 public:
  explicit PredictionError(const std::string& message)
      : FastTextThreadedError(message) {}
};

class InvalidInputError : public FastTextThreadedError {
 public:
  explicit InvalidInputError(const std::string& message)
      : FastTextThreadedError(message) {}
};

}  // namespace fasttext_threaded
