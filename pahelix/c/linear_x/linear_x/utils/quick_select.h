#ifndef LINEAR_X_UTILS_QUICK_SELECT_H
#define LINEAR_X_UTILS_QUICK_SELECT_H

#include <vector>

template<typename ValueType>
unsigned long quickselect_partition(
        std::vector<std::pair<ValueType, int>>& scores,
        unsigned long lower,
        unsigned long upper) {
    float pivot = scores[upper].first;
    while (lower < upper) {
        while (scores[lower].first < pivot) {
            ++lower;
        }
        while (scores[upper].first > pivot) {
            --upper;
        }
        if (scores[lower].first == scores[upper].first) ++lower;
        else if (lower < upper) swap(scores[lower], scores[upper]);
    }
    return upper;
}

// in-place quick-select
template<typename ValueType>
ValueType quickselect(
        std::vector<std::pair<ValueType, int>>& scores,
        unsigned long lower,
        unsigned long upper,
        unsigned long k) {
    if ( lower == upper ) {
        return scores[lower].first;
    }
    unsigned long split = quickselect_partition(scores, lower, upper);
    unsigned long length = split - lower + 1;
    if (length == k) {
        return scores[split].first;
    } else if (k  < length) {
        return quickselect(scores, lower, split-1, k);
    } else {
        return quickselect(scores, split+1, upper, k - length);
    }
}


#endif // LINEAR_X_UTILS_QUICK_SELECT_H
