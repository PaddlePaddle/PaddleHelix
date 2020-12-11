#ifndef LINEAR_X_LINEAR_FOLD_BEAM_CKY_PARSER_H
#define LINEAR_X_LINEAR_FOLD_BEAM_CKY_PARSER_H

#include <string>
#include <limits>
#include <vector>
#include <unordered_map>
#include "utils/utility.h"
#include "utils/utility_v.h"
#include "linear_fold/utility.h"
#include "linear_fold/state.h"
#include "utils/quick_select.h"

#define MIN_CUBE_PRUNING_SIZE 20

template<typename ValueType>
class LinearFoldBeamCKYParser {
public:
    int beam_size;
    bool no_sharp_turn;
    bool use_constraints; // add constraints
    ValueType value_min;

    LinearFoldBeamCKYParser(
            int beam_size,
            bool use_constraints,
            bool no_sharp_turn);

    DecoderResult<ValueType> parse(std::string& seq, std::vector<int>& cons);

protected:
    unsigned seq_length;

    std::vector<std::unordered_map<int, LinearFoldState<ValueType>>> bestH, bestP, bestM2, bestMulti, bestM;

    std::vector<int> if_tetraloops;
    std::vector<int> if_hexaloops;
    std::vector<int> if_triloops;

    // same as bestM, but ordered
    std::vector<std::vector<std::pair<ValueType, int>>> sorted_bestM;

    // hzhang: sort keys in each beam to avoid randomness
    std::vector<std::pair<int, LinearFoldState<ValueType>>> keys;

    // vector to store the scores at each beam temporarily for beam pruning
    std::vector<std::pair<int, int>> scores;

    std::vector<LinearFoldState<ValueType>> bestC;

    std::vector<int> nucs;

    // lisiz: constraints
    std::vector<int> allow_unpaired_position;
    std::vector<int> allow_unpaired_range;

protected:
    void prepare(unsigned len);

    void post_process();

    bool get_parentheses(char* result, std::string& seq);

    // hzhang: sort keys in each beam to avoid randomness
    void sort_keys(std::unordered_map<int, LinearFoldState<ValueType>>& map, std::vector<std::pair<int, LinearFoldState<ValueType>>>& sorted_keys);

    void sort_m(ValueType threshold,
               std::unordered_map<int, LinearFoldState<ValueType>>& beamstep,
               std::vector<std::pair<ValueType, int>>& sorted_stepM);

    bool allow_paired(int i, int j, std::vector<int>& cons, char nuci, char nucj);

    void update_if_better(LinearFoldState<ValueType>& state, ValueType newscore, Manner manner) {
        if (state.score < newscore) {
            state.set(newscore, manner);
        }
    };

    void update_if_better(LinearFoldState<ValueType>& state, ValueType newscore, Manner manner, int split) {
        if (state.score < newscore || state.manner == MANNER_NONE) {
            state.set(newscore, manner, split);
        }
    };

    void update_if_better(LinearFoldState<ValueType>& state, ValueType newscore, Manner manner, char l1, int l2) {
        if (state.score < newscore || state.manner == MANNER_NONE) {
            state.set(newscore, manner, l1, l2);
        }
    };

    ValueType beam_prune(std::unordered_map<int, LinearFoldState<ValueType>>& beamstep);
};

template<typename ValueType>
LinearFoldBeamCKYParser<ValueType>::LinearFoldBeamCKYParser(
        int beam_size,
        bool use_constraints,
        bool no_sharp_turn) :
        beam_size(beam_size),
        use_constraints(use_constraints),
        no_sharp_turn(no_sharp_turn) {
    value_min = std::numeric_limits<ValueType>::lowest();
}

template<typename ValueType>
void LinearFoldBeamCKYParser<ValueType>::prepare(unsigned len) {
    seq_length = len;

    bestH.clear();
    bestH.resize(seq_length);
    bestP.clear();
    bestP.resize(seq_length);
    bestM2.clear();
    bestM2.resize(seq_length);
    bestM.clear();
    bestM.resize(seq_length);
    bestC.clear();
    bestC.resize(seq_length);
    bestMulti.clear();
    bestMulti.resize(seq_length);

    sorted_bestM.clear();
    sorted_bestM.resize(seq_length);

    keys.clear();
    keys.resize(seq_length);

    nucs.clear();
    nucs.resize(seq_length);

    scores.clear();
    scores.reserve(seq_length);

    if (use_constraints) {
        allow_unpaired_position.clear();
        allow_unpaired_position.resize(seq_length);

        allow_unpaired_range.clear();
        allow_unpaired_range.resize(seq_length);
    }
}

template<typename ValueType>
void LinearFoldBeamCKYParser<ValueType>::post_process() {
    bestH.clear();
    bestP.clear();
    bestM2.clear();
    bestM.clear();
    bestC.clear();
    bestMulti.clear();

    sorted_bestM.clear();
    keys.clear();
    nucs.clear();
    scores.clear();

    if (use_constraints) {
        allow_unpaired_position.clear();
        allow_unpaired_range.clear();
    }
}

template<typename ValueType>
bool LinearFoldBeamCKYParser<ValueType>::get_parentheses(char* result, std::string& seq) {
    memset(result, '.', seq_length);
    result[seq_length] = 0;

    std::stack<std::tuple<int, int, LinearFoldState<ValueType>>> stk;
    stk.push(std::make_tuple(0, seq_length-1, bestC[seq_length-1]));

    std::vector<std::pair<int,int>> multi_todo;
    std::unordered_map<int,int> mbp; // multi bp
    double total_energy = .0;
    double external_energy = .0;

    while ( !stk.empty() ) {
        std::tuple<int, int, LinearFoldState<ValueType>> top = stk.top();
        int i = std::get<0>(top), j = std::get<1>(top);
        LinearFoldState<ValueType>& state = std::get<2>(top);
        stk.pop();

        switch (state.manner) {
            case MANNER_H:
                // this state should not be traced
                break;
            case MANNER_HAIRPIN:
                {
                    result[i] = '(';
                    result[j] = ')';
                }
                break;
            case MANNER_SINGLE:
                {
                    result[i] = '(';
                    result[j] = ')';
                    int p = i + state.trace.paddings.l1;
                    int q = j - state.trace.paddings.l2;
                    stk.push(std::make_tuple(p, q, bestP[q][p]));
                }
                break;
            case MANNER_HELIX:
                {
                    result[i] = '(';
                    result[j] = ')';
                    stk.push(std::make_tuple(i+1, j-1, bestP[j-1][i+1]));
                }
                break;
            case MANNER_MULTI:
                {
                    int p = i + state.trace.paddings.l1;
                    int q = j - state.trace.paddings.l2;
                    stk.push(std::make_tuple(p, q, bestM2[q][p]));
                }
                break;
            case MANNER_MULTI_EQ_MULTI_PLUS_U:
                {
                    int p = i + state.trace.paddings.l1;
                    int q = j - state.trace.paddings.l2;
                    stk.push(std::make_tuple(p, q, bestM2[q][p]));
                }
                break;
            case MANNER_P_EQ_MULTI:
                {
                    result[i] = '(';
                    result[j] = ')';
                    stk.push(std::make_tuple(i, j, bestMulti[j][i]));
                }
                break;
            case MANNER_M2_EQ_M_PLUS_P:
                {
                     int k = state.trace.split;
                    stk.push(std::make_tuple(i, k, bestM[k][i]));
                    stk.push(std::make_tuple(k+1, j, bestP[j][k+1]));
                }
                break;
            case MANNER_M_EQ_M2:
                stk.push(std::make_tuple(i, j, bestM2[j][i]));
                break;
            case MANNER_M_EQ_M_PLUS_U:
                stk.push(std::make_tuple(i, j-1, bestM[j-1][i]));
                break;
            case MANNER_M_EQ_P:
                stk.push(std::make_tuple(i, j, bestP[j][i]));
                break;
            case MANNER_C_EQ_C_PLUS_U:
                {
                    int k = j - 1;
                    if (k != -1)
                        stk.push(std::make_tuple(0, k, bestC[k]));
                }
                break;
            case MANNER_C_EQ_C_PLUS_P:
                {
                    int k = state.trace.split;
                    if (k != -1) {
                        stk.push(std::make_tuple(0, k, bestC[k]));
                        stk.push(std::make_tuple(k+1, j, bestP[j][k+1]));
                    }
                    else {
                        stk.push(std::make_tuple(i, j, bestP[j][i]));
                    }
                }
                break;
            default:  // MANNER_NONE or other cases
                if (use_constraints) {
                    printf("We can't find a valid structure for this sequence and constraint.\n");
                    printf("There are two minor restrictions in our real system:\n");
                    printf("the length of an interior loop is bounded by 30nt \n");
                    printf("(a standard limit found in most existing RNA folding software such as CONTRAfold)\n");
                    printf("so is the leftmost (50-end) unpaired segment of a multiloop (new constraint).\n");
                    return false;
                } 
                printf("wrong manner at %d, %d: manner %d\n", i, j, state.manner); fflush(stdout);
                assert(false);
                
        }
    }

    return true;
}

template<typename ValueType>
void LinearFoldBeamCKYParser<ValueType>::sort_m(
        ValueType threshold,
        std::unordered_map<int, LinearFoldState<ValueType>>& beamstep,
        std::vector<std::pair<ValueType, int>>& sorted_stepM) {
    sorted_stepM.clear();
    if (abs(threshold - value_min) < 1e-7) {
        // no beam_size pruning before, so scores vector not usable
        for (auto &item : beamstep) {
            int i = item.first;
            LinearFoldState<ValueType> &cand = item.second;
            int k = i - 1;
            double newscore = 0.0;
            // lisiz: constraints may cause all VALUE_MIN, sorting has no use
            if ((use_constraints) && (k >= 0) && (bestC[k].score == value_min)) newscore = cand.score;
            else newscore = (k >= 0 ? bestC[k].score : 0) + cand.score;
            sorted_stepM.push_back(std::make_pair(newscore, i));
        }
    } else {
        for (auto &p : scores) {
            if (p.first >= threshold) sorted_stepM.push_back(p);
        }
    }

    sort(sorted_stepM.begin(), sorted_stepM.end(), std::greater<std::pair<double, int>>());
}

// lisiz, constraints
template<typename ValueType>
bool LinearFoldBeamCKYParser<ValueType>::allow_paired(
        int i,
        int j,
        std::vector<int>& cons,
        char nuci,
        char nucj) {
    return (cons[i] == -1 || cons[i] == j) && (cons[j] == -1 || cons[j] == i) && g_allowed_pairs[nuci][nucj];
}

template<typename ValueType>
ValueType LinearFoldBeamCKYParser<ValueType>::beam_prune(std::unordered_map<int, LinearFoldState<ValueType>> &beamstep) {
    scores.clear();
    for (auto &item : beamstep) {
        int i = item.first;
        LinearFoldState<ValueType>& cand = item.second;
        int k = i - 1;
        double newscore = 0.0;
        // lisiz: for _V, avoid -inf-int=+inf
        if ((k >= 0) && (bestC[k].score == value_min)) newscore = value_min;
        else newscore = (k >= 0 ? bestC[k].score : 0) + cand.score;
        scores.push_back(std::make_pair(newscore, i));
    }
    if (scores.size() <= beam_size) return value_min;
    double threshold = quickselect(scores, 0, scores.size() - 1, scores.size() - beam_size);
    for (auto &p : scores) {
        if (p.first < threshold) beamstep.erase(p.second);
    }

    return threshold;
}

DecoderResult<double> linear_fold(
        std::string& sequence,
        int beam_size,
        char energy_model,
        bool use_constraints,
        std::vector<int>& constraints,
        bool no_sharp_turn);

#endif //LINEAR_X_LINEAR_FOLD_BEAM_CKY_PARSER_H
