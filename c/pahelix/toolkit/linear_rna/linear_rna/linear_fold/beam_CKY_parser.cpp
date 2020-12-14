#include "linear_fold/beam_CKY_parser.h"
#include "linear_fold/beam_CKY_parser_c.h"
#include "linear_fold/beam_CKY_parser_v.h"

DecoderResult<double> linear_fold(
        std::string& sequence,
        int beam_size,
        char energy_model,
        bool use_constraints,
        std::vector<int>& constraints,
        bool no_sharp_turn) {

    if ((energy_model != 'c') && (energy_model != 'v')){
        printf("energy model should be either 'c' (CONTRAfold, by default) or 'v' (Vinnea)\n");
        return {"", 0.0};
    }

    if (energy_model == 'c'){
        LinearFoldBeamCKYParserC parser(beam_size, use_constraints, no_sharp_turn);
        return parser.parse(sequence, constraints);
    } else {
        LinearFoldBeamCKYParserV parser(beam_size, use_constraints, no_sharp_turn);
        return parser.parse(sequence, constraints);
    }
}
