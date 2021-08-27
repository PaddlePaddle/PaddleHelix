#ifndef LINEAR_RNA_LINEAR_RNA_H
#define LINEAR_RNA_LINEAR_RNA_H

#include <pybind11/pybind11.h>

#include "linear_partition/beam_CKY_parser.h"
#include "linear_fold/beam_CKY_parser.h"

namespace py = pybind11;

py::tuple linear_partition_c(
        std::string& sequence,
        int beam_size = 100,
        float bpp_cutoff = 0.0,
        bool no_sharp_turn = true);

py::tuple linear_partition_v(
        std::string& sequence,
        int beam_size = 100,
        float bpp_cutoff = 0.0,
        bool no_sharp_turn = true);

py::tuple linear_fold_c(
        std::string& sequence,
        int beam_size = 100,
        bool use_constraints = false,
        const std::string& constraint = "",
        bool no_sharp_turn = true);

py::tuple linear_fold_v(
        std::string& sequence,
        int beam_size = 100,
        bool use_constraints = false,
        const std::string& constraint = "",
        bool no_sharp_turn = true);

bool check_input(std::string& sequence);

bool check_input_references(
        const std::string& sequence,
        std::string& reference);

bool check_input_constraints(
        const std::string& sequence,
        const std::string& constraints_str,
        std::vector<int>& constraints_idx);

#endif // LINEAR_RNA_LINEAR_RNA_H
