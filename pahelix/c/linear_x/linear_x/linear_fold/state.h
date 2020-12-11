#ifndef LINEAR_X_LINEAR_FOLD_STATE_H
#define LINEAR_X_LINEAR_FOLD_STATE_H

template<typename ValueType>
struct LinearFoldState {
    ValueType score;
    Manner manner;

    union TraceInfo {
        int split;
        struct {
            char l1;
            int l2;
        } paddings;
    };

    TraceInfo trace;

    LinearFoldState(): manner(MANNER_NONE), score(std::numeric_limits<int>::lowest()) {};
    LinearFoldState(ValueType s, Manner m): score(s), manner(m) {};

    void set(ValueType score_, Manner manner_) {
        score = score_; manner = manner_;
    }

    void set(ValueType score_, Manner manner_, int split_) {
        score = score_; manner = manner_; trace.split = split_;
    }

    void set(ValueType score_, Manner manner_, char l1_, int l2_) {
        score = score_; manner = manner_;
        trace.paddings.l1 = l1_; trace.paddings.l2 = l2_;
    }
};

template<typename ValueType>
struct DecoderResult {
    std::string structure;
    ValueType score;
};

#endif // LINEAR_X_LINEAR_FOLD_STATE_H
