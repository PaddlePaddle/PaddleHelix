#ifndef LINEAR_X_LINEAR_FOLD_UTILITY_H
#define LINEAR_X_LINEAR_FOLD_UTILITY_H

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <stack>
#include <set>
#include <map>

enum Manner {
    MANNER_NONE = 0,              // 0: empty
    MANNER_H,                     // 1: hairpin candidate
    MANNER_HAIRPIN,               // 2: hairpin
    MANNER_SINGLE,                // 3: single
    MANNER_HELIX,                 // 4: helix
    MANNER_MULTI,                 // 5: multi = ..M2. [30 restriction on the left and jump on the right]
    MANNER_MULTI_EQ_MULTI_PLUS_U, // 6: multi = multi + U
    MANNER_P_EQ_MULTI,            // 7: P = (multi)
    MANNER_M2_EQ_M_PLUS_P,        // 8: M2 = M + P
    MANNER_M_EQ_M2,               // 9: M = M2
    MANNER_M_EQ_M_PLUS_U,         // 10: M = M + U
    MANNER_M_EQ_P,                // 11: M = P
    /* MANNER_C_eq_U, */
    /* MANNER_C_eq_P, */
    MANNER_C_EQ_C_PLUS_U,     // 12: C = C + U
    MANNER_C_EQ_C_PLUS_P,     // 13: C = C + P
};

#endif  // LINEAR_X_LINEAR_FOLD_UTILITY_H
