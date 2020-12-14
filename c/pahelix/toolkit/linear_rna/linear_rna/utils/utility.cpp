#include "utils/utility.h"

bool g_allowed_pairs[NOTON][NOTON];
bool g_allowed_helix_stacking[NOTON][NOTON][NOTON][NOTON];
float g_cache_single[SINGLE_MAX_LEN+1][SINGLE_MAX_LEN+1];

void initialize_cachesingle() {
    memset(g_cache_single, 0, sizeof(g_cache_single));
    for (int l1 = SINGLE_MIN_LEN; l1 <= SINGLE_MAX_LEN; l1++) {
        for (int l2 = SINGLE_MIN_LEN; l2 <= SINGLE_MAX_LEN; l2++) {
            if (l1 == 0 && l2 == 0){
                continue;
            }
            // bulge
            else if (l1 == 0){
                g_cache_single[l1][l2] += g_bulge_length[l2];
            }
            else if (l2 == 0){
                g_cache_single[l1][l2] += g_bulge_length[l1];
            }
            else
            {
                // internal
                g_cache_single[l1][l2] += g_internal_length[std::min(l1 + l2, INTERNAL_MAX_LEN)];

                // internal explicit
                if (l1 <= EXPLICIT_MAX_LEN && l2 <= EXPLICIT_MAX_LEN){
                    g_cache_single[l1][l2] +=
                            g_internal_explicit[l1 <= l2 ? l1 * EXPLICIT_MAX_LEN + l2 : l2 * EXPLICIT_MAX_LEN + l1];
                }
                // internal symmetry
                if (l1 == l2)
                {
                    g_cache_single[l1][l2] += g_internal_symmetric_length[std::min(l1, SYMMETRIC_MAX_LEN)];
                }   
                else {  // internal asymmetry
                    int diff = l1 - l2; 
                    if (diff < 0) {
                        diff = -diff;
                    }
                    g_cache_single[l1][l2] += g_internal_asymmetry[std::min(diff, ASYMMETRY_MAX_LEN)];
                }
            }
        }
    }
    return;
}

void initialize()
{
    g_allowed_pairs[GET_ACGU_NUM('A')][GET_ACGU_NUM('U')] = true;
    g_allowed_pairs[GET_ACGU_NUM('U')][GET_ACGU_NUM('A')] = true;
    g_allowed_pairs[GET_ACGU_NUM('C')][GET_ACGU_NUM('G')] = true;
    g_allowed_pairs[GET_ACGU_NUM('G')][GET_ACGU_NUM('C')] = true;
    g_allowed_pairs[GET_ACGU_NUM('G')][GET_ACGU_NUM('U')] = true;
    g_allowed_pairs[GET_ACGU_NUM('U')][GET_ACGU_NUM('G')] = true;

    helix_stacking_old('A', 'U', 'A', 'U'); // = true;
    helix_stacking_old('A', 'U', 'C', 'G'); // = true;
    helix_stacking_old('A', 'U', 'G', 'C'); // = true;
    helix_stacking_old('A', 'U', 'G', 'U'); // = true;
    helix_stacking_old('A', 'U', 'U', 'A'); // = true;
    helix_stacking_old('A', 'U', 'U', 'G'); // = true;
    helix_stacking_old('C', 'G', 'A', 'U'); // = true;
    helix_stacking_old('C', 'G', 'C', 'G'); // = true;
    helix_stacking_old('C', 'G', 'G', 'C'); // = true;
    helix_stacking_old('C', 'G', 'G', 'U'); // = true;
    helix_stacking_old('C', 'G', 'U', 'G'); // = true;
    helix_stacking_old('G', 'C', 'A', 'U'); // = true;
    helix_stacking_old('G', 'C', 'C', 'G'); // = true;
    helix_stacking_old('G', 'C', 'G', 'U'); // = true;
    helix_stacking_old('G', 'C', 'U', 'G'); // = true;
    helix_stacking_old('G', 'U', 'A', 'U'); // = true;
    helix_stacking_old('G', 'U', 'G', 'U'); // = true;
    helix_stacking_old('G', 'U', 'U', 'G'); // = true;
    helix_stacking_old('U', 'A', 'A', 'U'); // = true;
    helix_stacking_old('U', 'A', 'G', 'U'); // = true;
    helix_stacking_old('U', 'G', 'G', 'U'); // = true;
}

