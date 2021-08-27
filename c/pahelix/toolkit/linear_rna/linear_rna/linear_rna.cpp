#include "linear_rna.h"

#include <algorithm>
#include <stdio.h> 
#include <map>
#include <stack>

#include "utils/utility.h"
#include "linear_fold/beam_CKY_parser_c.h"
#include "linear_fold/beam_CKY_parser_v.h"

py::tuple linear_partition_c(
        std::string& sequence,
        int beam_size,
        float bpp_cutoff,
        bool no_sharp_turn) {

    py::list result_list;
    float score = 0.0; // TO DO
    if (check_input(sequence)) {
        LinearPartitionBeamCKYParser parser(beam_size, 'c', no_sharp_turn, bpp_cutoff);
        score = parser.parse(sequence);
        const std::unordered_map<std::pair<int, int>, float, hash_pair>& pij = parser.get_pij();
        int seq_len = sequence.length();
        int turn = no_sharp_turn ? 3 : 0;
        for (int i = 1; i <= seq_len; ++i) {
            for (int j = i + turn + 1; j <= seq_len; j++) {
                std::pair<int, int> key = std::make_pair(i, j);
                auto iter = pij.find(key);
                if (iter != pij.end()) {
                    py::tuple result = py::make_tuple(i, j, iter->second);
                    result_list.append(result);
                }
            }
        }
        parser.post_process();
    }
    return py::make_tuple(score, result_list);
}

py::tuple linear_partition_v(
        std::string& sequence,
        int beam_size,
        float bpp_cutoff,
        bool no_sharp_turn) {

    py::list result_list;
    float score = 0.0; // TO DO
    if (check_input(sequence)) {
        LinearPartitionBeamCKYParser parser(beam_size, 'v', no_sharp_turn, bpp_cutoff);
        score = parser.parse(sequence);
        const std::unordered_map<std::pair<int, int>, float, hash_pair>& pij = parser.get_pij();
        int seq_len = sequence.length();
        int turn = no_sharp_turn ? 3 : 0;
        for (int i = 1; i <= seq_len; ++i) {
            for (int j = i + turn + 1; j <= seq_len; j++) {
                std::pair<int, int> key = std::make_pair(i, j);
                auto iter = pij.find(key);
                if (iter != pij.end()) {
                    py::tuple result = py::make_tuple(i, j, iter->second);
                    result_list.append(result);
                }
            }
        }
        parser.post_process();
    }
    return py::make_tuple(score, result_list);
}

py::tuple linear_fold_c(
        std::string& sequence,
        int beam_size,
        bool use_constraints,
        const std::string& constraint,
        bool no_sharp_turn) {
    // check input
    if (!check_input(sequence)) {
        return py::make_tuple("", 0);
    }
    // check constraints
    std::vector<int> constraints_idx;
    if (use_constraints) {
        if (!check_input_constraints(sequence, constraint, constraints_idx)) {
            return py::make_tuple("", 0);
        }
    }

    DecoderResult<double> result = linear_fold(sequence, beam_size, 'c', use_constraints, constraints_idx, no_sharp_turn);
    return py::make_tuple(result.structure, result.score);
}

py::tuple linear_fold_v(
        std::string& sequence,
        int beam_size,
        bool use_constraints,
        const std::string& constraint,
        bool no_sharp_turn) {
    // check input
    if (!check_input(sequence)) {
        return py::make_tuple("", 0);
    }
    // check constraints
    std::vector<int> constraints_idx;
    if (use_constraints) {
        if (!check_input_constraints(sequence, constraint, constraints_idx)) {
            return py::make_tuple("", 0);
        }
    }

    DecoderResult<double> result = linear_fold(sequence, beam_size, 'v', use_constraints, constraints_idx, no_sharp_turn);
    return py::make_tuple(result.structure, result.score);
}

bool check_input(std::string& sequence) {
    // convert to uppercase
    std::transform(sequence.begin(), sequence.end(), sequence.begin(), ::toupper);
    // convert T to U
    std::replace(sequence.begin(), sequence.end(), 'T', 'U');
    int length = sequence.length();
    for (int i = 0; i < length; i++) {
        if (!std::isalpha(sequence[i])) {
            printf("Unrecognized character in %s\n, should be alphabetic", sequence.c_str());
            return false;
        }
    }
    return true;
}

bool check_input_references(
        const std::string& sequence,
        std::string& reference) {
    if (reference.length() != sequence.length()) {
        printf("the length of the reference structure should have the same length as the corresponding input sequence");
        return false;
    }
    // remove peudoknots
    char r = '\0';
    std::map<char, char> rs = { {'[', '.'}, {']', '.'}, {'{', '.'}, {'}', '.'}, {'<', '.'}, {'>', '.'} };
    replace_if(reference.begin(), reference.end(), [&](char c) { return r = rs[c]; }, r);

    // check reference
    int n = reference.length();
    for (int i = 0; i < n; i++) {
        char coni = reference[i];
        if ((coni != '.') && (coni != '(') && (coni != ')')) {
            printf("Unrecognized structure character in %s\n, should be . ( or ) \n", reference.c_str());
            return false;
        }
    }
    return true;
}

bool check_input_constraints(
        const std::string& sequence,
        const std::string& constraints_str,
        std::vector<int>& constraints_idx) {
    if (constraints_str.length() != sequence.length()) {
        printf("the length of the constraint string should have the same length as the corresponding input sequence");
        return false;
    }

    int n = constraints_str.length();
    constraints_idx = std::vector<int>(n);
    std::stack<int> leftBrackets;
    for (int i = 0; i < n; i++) {
        char coni = constraints_str[i];
        if ((coni != '.') && (coni != '(') && (coni != ')') && (coni != '?')) {
            printf("Unrecognized constraint character in %s\n, should be ? . ( or )\n",
                    constraints_str.c_str());
            return false;
        }
        switch(coni) {
            case '.':
                constraints_idx[i] = -2;
                break;
            case '?':
                constraints_idx[i] = -1;
                break;
            case '(':
                leftBrackets.push(i);
                break;
            case ')':
                {
                    int leftIndex = leftBrackets.top();
                    leftBrackets.pop();
                    char pair[3] = {sequence[i], sequence[leftIndex], '\0'};
                    if (strcmp(pair, "CG") * strcmp(pair, "GC") * strcmp(pair, "AU") *
                            strcmp(pair, "UA") * strcmp(pair, "GU") * strcmp(pair, "UG") != 0) {
                        printf("Constrains on non-classical base pairs (non AU, CG, GU pairs)\n");
                        return false;
                    }
                    constraints_idx[leftIndex] = i;
                    constraints_idx[i] = leftIndex;
                }
                break;
        }
    }
    return true;
}

PYBIND11_MODULE(linear_rna, m) {
    initialize();
    initialize_cachesingle();
    m.def("linear_partition_c",
            &linear_partition_c,
            py::arg("sequence") = "",
            py::arg("beam_size") = 100,
            py::arg("bp_cutoff") = 0.0,
            py::arg("no_sharp_turn") = true,
            py::return_value_policy::reference);
    m.def("linear_partition_v",
            &linear_partition_v,
            py::arg("sequence") = "",
            py::arg("beam_size") = 100,
            py::arg("bp_cutoff") = 0.0,
            py::arg("no_sharp_turn") = true,
            py::return_value_policy::reference);
    m.def("linear_fold_c",
            &linear_fold_c,
            py::arg("sequence") = "",
            py::arg("beam_size") = 100,
            py::arg("use_constraints") = false,
            py::arg("constraint") = "",
            py::arg("no_sharp_turn") = true,
            py::return_value_policy::reference);
    m.def("linear_fold_v",
            &linear_fold_v,
            py::arg("sequence") = "",
            py::arg("beam_size") = 100,
            py::arg("use_constraints") = false,
            py::arg("constraint") = "",
            py::arg("no_sharp_turn") = true,
            py::return_value_policy::reference);
}
