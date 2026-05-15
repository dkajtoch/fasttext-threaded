#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "fasttext_parallel/errors.h"
#include "fasttext_parallel/parallel_model.h"

namespace py = pybind11;

namespace fasttext_parallel {
namespace {

py::tuple predictions_to_python(const std::vector<PredictionResult>& results) {
  py::list all_labels;
  py::list all_probabilities;

  for (const PredictionResult& result : results) {
    py::list labels;
    for (const std::string& label : result.labels) {
      labels.append(py::str(label));
    }

    py::array_t<fasttext::real> probabilities(result.probabilities.size());
    auto mutable_probabilities = probabilities.mutable_unchecked<1>();
    for (py::ssize_t index = 0;
         index < static_cast<py::ssize_t>(result.probabilities.size()); ++index) {
      mutable_probabilities(index) =
          result.probabilities[static_cast<std::size_t>(index)];
    }

    all_labels.append(labels);
    all_probabilities.append(probabilities);
  }

  return py::make_tuple(all_labels, all_probabilities);
}

}  // namespace
}  // namespace fasttext_parallel

PYBIND11_MODULE(_fasttext_parallel, module) {
  module.doc() = "Native extension for fasttext-parallel";

  auto base_error = py::register_exception<fasttext_parallel::FastTextParallelError>(
      module, "FastTextParallelError", PyExc_RuntimeError);
  py::register_exception<fasttext_parallel::ModelLoadError>(module, "ModelLoadError",
                                                            base_error.ptr());
  py::register_exception<fasttext_parallel::PredictionError>(module, "PredictionError",
                                                             base_error.ptr());
  py::register_exception<fasttext_parallel::InvalidInputError>(
      module, "InvalidInputError", base_error.ptr());

  py::class_<fasttext_parallel::ParallelModel>(module, "FastTextParallelModel")
      .def(py::init<const std::string&, std::size_t>(), py::arg("model_path"),
           py::arg("threads"), py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("threads", &fasttext_parallel::ParallelModel::threads)
      .def(
          "predict_many",
          [](fasttext_parallel::ParallelModel& model,
             const std::vector<std::string>& lines, int32_t k, fasttext::real threshold,
             const char* /*on_unicode_error*/) {
            std::vector<fasttext_parallel::PredictionResult> results;
            {
              py::gil_scoped_release release;
              results = model.predict_many(lines, k, threshold);
            }
            return fasttext_parallel::predictions_to_python(results);
          },
          py::arg("lines"), py::arg("k"), py::arg("threshold"),
          py::arg("on_unicode_error") = "strict");
}
