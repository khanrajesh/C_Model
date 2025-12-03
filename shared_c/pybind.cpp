#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Reuse existing C++ implementation
#include "../shared_c/model/supervised/linear_regeression/LinearRegression.cpp"
#include "../shared_c/model/supervised/logistic_regeression/LogisticRegression.cpp"


namespace py = pybind11;

PYBIND11_MODULE(shared_c_ext, m) {
    py::class_<LinearRegression>(m, "LinearRegression")
        .def(py::init<double>(), py::arg("learning_rate") = 0.000001)
        .def(
            "fit",
            &LinearRegression::fit,
            py::arg("X"),
            py::arg("y"),
            py::arg("epochs") = 1000)
        // Batch predict helper: accepts a list of feature rows and returns a list of predictions
        .def(
            "predict",
            [](const LinearRegression& model, const std::vector<std::vector<double>>& X) {
                std::vector<double> out;
                out.reserve(X.size());
                for (const auto& row : X) {
                    out.push_back(model.predict(row));
                }
                return out;
            },
            py::arg("X"))
        .def("coefficients", &LinearRegression::coefficeients);

    py::class_<LogisticRegression>(m, "LogisticRegression")
        .def(py::init<>())
        .def("fit",
             [](LogisticRegression& self,
                const std::vector<std::vector<double>>& X,
                const std::vector<double>& y,
                int epochs) {
                 std::vector<double> y_copy = y;
                 self.fit(X, y_copy, epochs);
             },
             py::arg("X"),
             py::arg("y"),
             py::arg("epochs") = 1000)
        .def("predict",
             &LogisticRegression::predict,
             py::arg("x"));
}

