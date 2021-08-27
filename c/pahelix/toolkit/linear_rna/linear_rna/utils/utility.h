#ifndef LINEAR_RNA_UTILS_UTILITY_H
#define LINEAR_RNA_UTILS_UTILITY_H

#include <algorithm>
#include <cstring>
#include <assert.h>

#include "feature_weight.h"

#define INF 1000000007

#define NOTON 5 // NUM_OF_TYPE_OF_NUCS
#define NOTOND 25
#define NOTONT 125

#define EXPLICIT_MAX_LEN 4
#define SINGLE_MIN_LEN 0
#define SINGLE_MAX_LEN 30  // NOTE: *must* <= sizeof(char), otherwise modify State::TraceInfo accordingly

#define MULTI_MAX_LEN 30

#define HAIRPIN_MAX_LEN 30
#define BULGE_MAX_LEN SINGLE_MAX_LEN
#define INTERNAL_MAX_LEN SINGLE_MAX_LEN
#define SYMMETRIC_MAX_LEN 15
#define ASYMMETRY_MAX_LEN 28

#define GET_ACGU_NUM(x) ((x=='A'? 0 : (x=='C'? 1 : (x=='G'? 2 : (x=='U'?3: 4)))))

extern bool g_allowed_pairs[NOTON][NOTON];
extern bool g_allowed_helix_stacking[NOTON][NOTON][NOTON][NOTON];
extern float g_cache_single[SINGLE_MAX_LEN+1][SINGLE_MAX_LEN+1];

inline void helix_stacking_old(char x, char y, char z, char w){
    g_allowed_helix_stacking[GET_ACGU_NUM(x)][GET_ACGU_NUM(y)][GET_ACGU_NUM(z)][GET_ACGU_NUM(w)] = true;
}

void initialize_cachesingle();

void initialize();

// ------------- nucs based scores -------------

// parameters: nucs[i], nucs[j]
inline float base_pair_score(int nuci, int nucj) {
    return g_base_pair[nucj*NOTON + nuci];
}

// parameters: nucs[i], nucs[i+1], nucs[j-1], nucs[j]
inline float helix_stacking_score(int nuci, int nuci1, int nucj_1, int nucj) {
    return g_helix_stacking[nuci*NOTONT + nucj*NOTOND + nuci1*NOTON + nucj_1];
}

// parameters: nucs[i], nucs[j]
inline float helix_closing_score(int nuci, int nucj) {
    return g_helix_closing[nuci*NOTON + nucj];
}

// parameters: nucs[i], nucs[i+1], nucs[j-1], nucs[j]
inline float terminal_mismatch_score(int nuci, int nuci1, int nucj_1, int nucj) {
    return g_terminal_mismatch[nuci*NOTONT+nucj*NOTOND + nuci1*NOTON + nucj_1];
}


// parameter: nucs[i]
inline float bulge_nuc_score(int nuci) {
    return g_bulge_0x1_nucleotides[nuci];
}

// parameters: nucs[i], nucs[j]
inline float internal_nuc_score(int nuci, int nucj) {
    return g_internal_1x1_nucleotides[nuci*NOTON + nucj];
}

// parameters: nucs[i], nucs[i+1], nucs[j]
inline float dangle_left_score(int nuci, int nuci1, int nucj) {
    return g_dangle_left[nuci*NOTOND + nucj*NOTON + nuci1];
}

// parameters: nucs[i], nucs[j-1], nucs[j]
inline float dangle_right_score(int nuci, int nucj_1, int nucj) {
    return g_dangle_right[nuci*NOTOND + nucj*NOTON + nucj_1];
}



// ------------- length based scores -------------

inline float hairpin_score(int i, int j) {
    return g_hairpin_length[std::min(j-i-1, HAIRPIN_MAX_LEN)];
}

inline float internal_length_score(int l) {
    return g_internal_length[std::min(l, INTERNAL_MAX_LEN)];
}

inline float internal_explicit_score(int l1, int l2){
    int l1_ = std::min(l1, EXPLICIT_MAX_LEN);
    int l2_ = std::min(l2, EXPLICIT_MAX_LEN);
    return g_internal_explicit[l1_ <= l2_ ? l1_ * NOTON + l2_ : l2_ * NOTON + l1_];
}

inline float internal_sym_score(int l) {
    return g_internal_symmetric_length[std::min(l, SYMMETRIC_MAX_LEN)];
}

inline float internal_asym_score(int l1, int l2)
{
    int diff = l1 - l2; 
    if (diff < 0) {
        diff = -diff;
    }
    return g_internal_asymmetry[std::min(diff, ASYMMETRY_MAX_LEN)];
}

inline float bulge_length_score(int l){
    return g_bulge_length[std::min(l, BULGE_MAX_LEN)];
}

inline float hairpin_at_least_score(int l) {
    return g_hairpin_length_at_least[std::min(l, HAIRPIN_MAX_LEN)];
}

inline float buldge_length_at_least_score(int l) {
    return g_bulge_length_at_least[std::min(l, BULGE_MAX_LEN)];
}

inline float internal_length_at_least_score(int l) {
    return g_internal_length_at_least[std::min(l, INTERNAL_MAX_LEN)];
}


//-----------------------------------------------------
inline float score_junction_A(int i, int j, int nuci, int nuci1, int nucj_1, int nucj, int len) {
    return helix_closing_score(nuci, nucj) +
            (i < len - 1 ? dangle_left_score(nuci, nuci1, nucj) : 0) +
            (j > 0 ? dangle_right_score(nuci, nucj_1, nucj) : 0);
}

inline float score_junction_B(int i, int j, int nuci, int nuci1, int nucj_1, int nucj) {
    return helix_closing_score(nuci, nucj) + terminal_mismatch_score(nuci, nuci1, nucj_1, nucj);
}

inline float score_hairpin_length(int len) {
  return g_hairpin_length[std::min(len, HAIRPIN_MAX_LEN)];
}

inline float score_hairpin(int i, int j, int nuci, int nuci1, int nucj_1, int nucj) {
    return g_hairpin_length[std::min(j-i-1, HAIRPIN_MAX_LEN)] +
            score_junction_B(i, j, nuci, nuci1, nucj_1, nucj);
}

inline float score_helix(int nuci, int nuci1, int nucj_1, int nucj) {
    return helix_stacking_score(nuci, nuci1, nucj_1, nucj) + base_pair_score(nuci1, nucj_1);
}

inline float score_single_nuc(int i, int j, int p, int q, int nucp_1, int nucq1) {
    int l1 = p-i-1, l2=j-q-1;
    if (l1==0 && l2==1) return bulge_nuc_score(nucq1);
    if (l1==1 && l2==0) return bulge_nuc_score(nucp_1);
    if (l1==1 && l2==1) return internal_nuc_score(nucp_1, nucq1);
    return 0;
}

inline float score_single(int i, int j, int p, int q, int len,
                           int nuci, int nuci1, int nucj_1, int nucj,
                           int nucp_1, int nucp, int nucq, int nucq1) {
    int l1 = p-i-1, l2=j-q-1;
    return g_cache_single[l1][l2] +
           base_pair_score(nucp, nucq) +
           score_junction_B(i, j, nuci, nuci1, nucj_1, nucj) +
           score_junction_B(q, p, nucq, nucq1, nucp_1, nucp) +
           score_single_nuc(i, j, p, q, nucp_1, nucq1);
}

// score_single without socre_junction_B
inline float score_single_without_junctionB(int i, int j, int p, int q,
                           int nucp_1, int nucp, int nucq, int nucq1) {
    int l1 = p-i-1, l2=j-q-1;
    return g_cache_single[l1][l2] +
           base_pair_score(nucp, nucq) +
           score_single_nuc(i, j, p, q, nucp_1, nucq1);
}

inline float score_multi(int i, int j, int nuci, int nuci1, int nucj_1, int nucj, int len) {
    return score_junction_A(i, j, nuci, nuci1, nucj_1, nucj, len) +
           g_multi_paired + g_multi_base;
}

inline float score_multi_unpaired(int i, int j) {
    return (j-i+1) * g_multi_unpaired;
}

inline float score_M1(int i, int j, int k, int nuci_1, int nuci, int nuck, int nuck1, int len) {
    return score_junction_A(k, i, nuck, nuck1, nuci_1, nuci, len) +
           score_multi_unpaired(k+1, j) + base_pair_score(nuci, nuck) + g_multi_paired;
}

inline float score_external_paired(int i, int j, int nuci_1, int nuci, int nucj, int nucj1, int len) {
    return score_junction_A(j, i, nucj, nucj1, nuci_1, nuci, len) +
           g_external_paired + base_pair_score(nuci, nucj);
}

inline float score_external_unpaired(int i, int j) {
    return (j-i+1) * g_external_unpaired;
}

#endif // LINEAR_RNA_UTILS_UTILITY_H
