#include "fasttext_parallel/parallel_model.h"

#include <algorithm>
#include <exception>
#include <future>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>

#include "fasttext_parallel/errors.h"

namespace fasttext_parallel {

namespace {

void validate_prediction_args(int32_t k) {
  if (k == 0 || k < -1) {
    throw InvalidInputError("k needs to be 1 or higher, or -1 for all labels");
  }
}

std::exception_ptr wrap_worker_exception(std::exception_ptr error) {
  try {
    if (error) {
      std::rethrow_exception(error);
    }
  } catch (const FastTextParallelError&) {
    return error;
  } catch (const std::exception& exc) {
    return std::make_exception_ptr(PredictionError(exc.what()));
  } catch (...) {
    return std::make_exception_ptr(PredictionError("unknown prediction error"));
  }
  return nullptr;
}

}  // namespace

ParallelModel::ParallelModel(const std::string& model_path, std::size_t thread_count)
    : pool_(thread_count) {
  if (model_path.empty()) {
    throw InvalidInputError("model_path must not be empty");
  }

  try {
    model_.loadModel(model_path);
  } catch (const std::exception& exc) {
    throw ModelLoadError("failed to load fastText model '" + model_path +
                         "': " + exc.what());
  } catch (...) {
    throw ModelLoadError("failed to load fastText model '" + model_path + "'");
  }
}

std::vector<PredictionResult> ParallelModel::predict_many(
    const std::vector<std::string>& lines, int32_t k, fasttext::real threshold) {
  validate_prediction_args(k);

  std::vector<PredictionResult> results(lines.size());
  if (lines.empty()) {
    return results;
  }

  const std::size_t task_count = std::min(pool_.size(), lines.size());
  const std::size_t chunk_size = (lines.size() + task_count - 1) / task_count;

  std::mutex error_mutex;
  std::exception_ptr first_error = nullptr;
  std::vector<std::future<void>> futures;
  futures.reserve(task_count);

  for (std::size_t task_index = 0; task_index < task_count; ++task_index) {
    const std::size_t begin = task_index * chunk_size;
    const std::size_t end = std::min(lines.size(), begin + chunk_size);
    if (begin >= end) {
      break;
    }

    futures.push_back(pool_.submit([&, begin, end]() {
      try {
        for (std::size_t index = begin; index < end; ++index) {
          results[index] = predict_one(lines[index], k, threshold);
        }
      } catch (...) {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (!first_error) {
          first_error = wrap_worker_exception(std::current_exception());
        }
      }
    }));
  }

  for (std::future<void>& future : futures) {
    future.get();
  }

  if (first_error) {
    std::rethrow_exception(first_error);
  }

  return results;
}

std::size_t ParallelModel::threads() const { return pool_.size(); }

PredictionResult ParallelModel::predict_one(const std::string& line, int32_t k,
                                            fasttext::real threshold) const {
  std::istringstream stream(line);
  std::vector<std::pair<fasttext::real, std::string>> predictions;
  model_.predictLine(stream, predictions, k, threshold);

  PredictionResult result;
  result.labels.reserve(predictions.size());
  result.probabilities.reserve(predictions.size());

  for (const auto& prediction : predictions) {
    result.probabilities.push_back(prediction.first);
    result.labels.push_back(prediction.second);
  }

  return result;
}

}  // namespace fasttext_parallel
