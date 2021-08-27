#ifndef LINEAR_RNA_LINEAR_FOLD_BEAM_CKY_PARSER_V_H
#define LINEAR_RNA_LINEAR_FOLD_BEAM_CKY_PARSER_V_H

#include "linear_fold/beam_CKY_parser.h"

bool g_compare_func(
        const std::pair<int, LinearFoldState<int>>& a,
        const std::pair<int, LinearFoldState<int>>& b);

class LinearFoldBeamCKYParserV : public LinearFoldBeamCKYParser<int> {
public:
    LinearFoldBeamCKYParserV(
            int beam_size,
            bool use_constraints = false,
            bool no_sharp_turn = false) :
        LinearFoldBeamCKYParser<int>(beam_size, use_constraints, no_sharp_turn) {}

    DecoderResult<double> parse(std::string& seq, std::vector<int>& cons);
protected:
    // hzhang: sort keys in each beam to avoid randomness
    void sort_keys(
            std::unordered_map<int, LinearFoldState<int>>& map,
            std::vector<std::pair<int, LinearFoldState<int>>>& sorted_keys);
};

#endif // LINEAR_RNA_LINEAR_FOLD_BEAM_CKY_PARSER_V_H
