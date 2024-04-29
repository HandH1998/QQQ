#include "w4a8/w4a8.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("w4a8_gemm", &w4a8_gemm, "INT8xINT4 matmul based marlin FP16xINT4 kernel.");
}