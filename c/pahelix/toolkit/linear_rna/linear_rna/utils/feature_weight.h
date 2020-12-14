#ifndef LINEAR_RNA_UTILS_FEATURE_WEIGHT_H
#define LINEAR_RNA_UTILS_FEATURE_WEIGHT_H

extern double g_multi_base;
extern double g_multi_unpaired;
extern double g_multi_paired;
extern double g_external_unpaired;
extern double g_external_paired;
extern double g_base_pair[25];
extern double g_internal_1x1_nucleotides[25];
extern double g_helix_stacking[625];
extern double g_terminal_mismatch[625];
extern double g_bulge_0x1_nucleotides[5];
extern double g_helix_closing[25];
extern double g_dangle_left[125];
extern double g_dangle_right[125];
extern double g_internal_explicit[21];
extern double g_hairpin_length[31];
extern double g_bulge_length[31];
extern double g_internal_length[31];
extern double g_internal_symmetric_length[16];
extern double g_internal_asymmetry[29];
extern double g_hairpin_length_at_least[31];
extern double g_bulge_length_at_least[31];
extern double g_internal_length_at_least[31];
extern double g_internal_symmetric_length_at_least[16];
extern double g_internal_asymmetry_at_least[29];
#endif // LINEAR_RNA_UTILS_FEATURE_WEIGHT_H
