#ifndef LINEAR_X_LINEAR_FOLD_BEAM_CKY_PARSER_C_H
#define LINEAR_X_LINEAR_FOLD_BEAM_CKY_PARSER_C_H

#include "linear_fold/beam_CKY_parser.h"

#define MIN_CUBE_PRUNING_SIZE 20

class LinearFoldBeamCKYParserC : public LinearFoldBeamCKYParser<double> {
public:
    LinearFoldBeamCKYParserC(
            int beam_size,
            bool use_constraints = false,
            bool no_sharp_turn = false) :
        LinearFoldBeamCKYParser<double>(beam_size, use_constraints, no_sharp_turn) {}

    DecoderResult<double> parse(std::string& seq, std::vector<int>& cons);
};

#endif // LINEAR_X_LINEAR_FOLD_BEAM_CKY_PARSER_C_H
