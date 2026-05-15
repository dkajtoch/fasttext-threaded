#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "fasttext_threaded/errors.h"
#include "fasttext_threaded/parallel_model.h"

int main() {
  try {
    fasttext_threaded::ParallelModel missing("/definitely/missing/model.bin", 2);
    (void)missing;
    assert(false);
  } catch (const fasttext_threaded::ModelLoadError&) {
  }

  const char* model_path = std::getenv("FASTTEXT_THREADED_TEST_MODEL");
  if (model_path == nullptr) {
    std::cerr << "FASTTEXT_THREADED_TEST_MODEL not set; skipping model test\n";
    return 0;
  }

  fasttext_threaded::ParallelModel model(model_path, 2);
  std::vector<std::string> lines = {"alpha beta\n", "gamma delta\n"};
  std::vector<fasttext_threaded::PredictionResult> results =
      model.predict_many(lines, 1, 0.0F);

  assert(results.size() == lines.size());
  for (const auto& result : results) {
    assert(result.labels.size() == result.probabilities.size());
  }

  try {
    (void)model.predict_many(lines, 0, 0.0F);
    assert(false);
  } catch (const fasttext_threaded::InvalidInputError&) {
  }

  return 0;
}
