#ifndef LINEAR_X_LINEAR_PARTITION_BEAM_CKY_PARSER_H
#define LINEAR_X_LINEAR_PARTITION_BEAM_CKY_PARSER_H

#include <string>
#include <limits>
#include <vector>
#include <unordered_map>
#include <math.h> 
#include "utils/fast_math.h"

// A hash function used to hash a pair of any kind 
struct hash_pair {
    template <class T1, class T2> 
    size_t operator()(const std::pair<T1, T2>& p) const { 
        auto hash1 = std::hash<T1>{}(p.first); 
        auto hash2 = std::hash<T2>{}(p.second); 
        return hash1 ^ hash2; 
    } 
};

struct LinearPartitionState {
    float alpha;
    float beta;
    LinearPartitionState(): alpha(VALUE_MIN), beta(VALUE_MIN) {};
};


class LinearPartitionBeamCKYParser {
public:
    int beam_size;
    char energy_model;
    bool no_sharp_turn;
    float bpp_cutoff;

    struct DecoderResult {
        float alpha;
        double time;
    };

    LinearPartitionBeamCKYParser(int beam_size=100,
              char energy_model='c',
              bool no_sharp_turn=true,
              float bpp_cutoff=0.0);

    float parse(std::string& seq);

    const std::unordered_map<std::pair<int, int>, float, hash_pair>& get_pij() const {
        return pij;
    }

    void post_process();

protected:
    unsigned seq_length;

    std::unordered_map<int, LinearPartitionState> *bestH, *bestP, *bestM2, *bestMulti, *bestM;

    std::vector<int> if_tetraloops;
    std::vector<int> if_hexaloops;
    std::vector<int> if_triloops;

    LinearPartitionState *bestC;

    int *nucs;

    std::vector<std::pair<float, int>> scores;
    std::unordered_map<std::pair<int, int>, float, hash_pair> pij;

protected:
    void prepare();

    void cal_pair_probs(LinearPartitionState& viterbi); 

    void outside(std::vector<int> next_pair[]);

    float beam_prune(std::unordered_map<int, LinearPartitionState>& beamstep);
};

#endif // LINEAR_X_LINEAR_PARTITION_BEAM_CKY_PARSER_H
