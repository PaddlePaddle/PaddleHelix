#include "linear_fold/beam_CKY_parser_c.h"

using namespace std;

DecoderResult<double> LinearFoldBeamCKYParserC::parse(string& seq, vector<int>& cons) {
    DecoderResult<double> decoder_result;

    // number of states
    unsigned long nos_H = 0, nos_P = 0, nos_M2 = 0,
            nos_M = 0, nos_C = 0, nos_Multi = 0;

    prepare(static_cast<unsigned>(seq.length()));

    for (int i = 0; i < seq_length; ++i) {
        nucs[i] = GET_ACGU_NUM(seq[i]);
    }
        
    // constraints
    if (use_constraints) {
        for (int i = 0; i < seq_length; i++) {
            int cons_idx = cons[i];
            allow_unpaired_position[i] = cons_idx == -1 || cons_idx == -2;
            if (cons_idx > -1) {
                if (!g_allowed_pairs[nucs[i]][nucs[cons_idx]]) {
                    printf("Constrains on non-classical base pairs (non AU, CG, GU pairs)\n");
                    return decoder_result;
                }
            }
        }
        int firstpair = seq_length;
        for (int i=seq_length-1; i>-1; i--) {
            allow_unpaired_range[i] = firstpair;
            if (cons[i] >= 0)
                firstpair = i;
        }
    }

    vector<int> next_pair[NOTON];
    {
        if (use_constraints) {
            for (int nuci = 0; nuci < NOTON; ++nuci) {
                next_pair[nuci].resize(seq_length, -1);
                int next = -1;
                for (int j = seq_length-1; j >=0; --j) {
                    next_pair[nuci][j] = next;
                    if (cons[j] > -2 && g_allowed_pairs[nuci][nucs[j]]) next = j;
                }
            }
        } else {
            for (int nuci = 0; nuci < NOTON; ++nuci) {
                next_pair[nuci].resize(seq_length, -1);
                int next = -1;
                for (int j = seq_length-1; j >=0; --j) {
                    next_pair[nuci][j] = next;
                    if (g_allowed_pairs[nuci][nucs[j]]) next = j;
                }
            }
        }
    }

    // start CKY decoding
    if (seq_length > 0) bestC[0].set(score_external_unpaired(0, 0), MANNER_C_EQ_C_PLUS_U);
    if (seq_length > 1) bestC[1].set(score_external_unpaired(0, 1), MANNER_C_EQ_C_PLUS_U);
    ++nos_C;

    // from left to right
    for(int j = 0; j < seq_length; ++j) {
        int nucj = nucs[j];
        int nucj1 = (j + 1) < seq_length ? nucs[j + 1] : -1;

        unordered_map<int, LinearFoldState<double>>& beamstepH = bestH[j];
        unordered_map<int, LinearFoldState<double>>& beamstepMulti = bestMulti[j];
        unordered_map<int, LinearFoldState<double>>& beamstepP = bestP[j];
        unordered_map<int, LinearFoldState<double>>& beamstepM2 = bestM2[j];
        unordered_map<int, LinearFoldState<double>>& beamstepM = bestM[j];
        LinearFoldState<double>& beamstepC = bestC[j];

        // beam_size of H
        {
            if (beam_size > 0 && beamstepH.size() > beam_size) beam_prune(beamstepH);

            {
                // for nucj put H(j, j_next) into H[j_next]
                int jnext = next_pair[nucj][j];
                if (no_sharp_turn) {
                    while (jnext - j < 4 && jnext != -1) {
                        jnext = next_pair[nucj][jnext];
                    }
                }

                // lisiz, constriants
                if (use_constraints) {
                    if (!allow_unpaired_position[j]) {
                        jnext = cons[j] > j ? cons[j] : -1; // lisiz: j must be left bracket, jump to the constrainted pair (j, j') directly
                    }
                    if (jnext != -1) {
                        int nucjnext = nucs[jnext];
                        if (jnext > allow_unpaired_range[j] || !allow_paired(j, jnext, cons, nucj, nucjnext))  // lisiz: avoid cross constrainted brackets or unallowed pairs
                            jnext = -1;
                    }
                }

                if (jnext != -1) {
                    int nucjnext = nucs[jnext];
                    int nucjnext_1 = (jnext - 1) > -1 ? nucs[jnext - 1] : -1;

                    double newscore = score_hairpin(j, jnext, nucj, nucj1, nucjnext_1, nucjnext);
                    // this candidate must be the best one at [j, jnext]
                    // so no need to check the score
                    update_if_better(bestH[jnext][j], newscore, MANNER_H);
                    ++ nos_H;
                }
            }

            {
                // for every state h in H[j]
                //   1. extend h(i, j) to h(i, jnext)
                //   2. generate p(i, j)
                for (auto &item : beamstepH) {
                    int i = item.first;
                    // printf("%d\n", i);
                    LinearFoldState<double> &state = item.second;
                    int nuci = nucs[i];
                    int jnext = next_pair[nuci][j];

                    // 2. generate p(i, j)
                    // lisiz, change the order because of the constriants
                    {
                        update_if_better(beamstepP[i], state.score, MANNER_HAIRPIN);
                        ++ nos_P;
                    }

                    // lisiz, constraints
                    if (jnext != -1 && use_constraints) {
                        int nucjnext = nucs[jnext];
                        if (jnext > allow_unpaired_range[i] || !allow_paired(i, jnext, cons, nuci, nucjnext))
                            continue;
                    }

                    if (jnext != -1) {
                        int nuci1 = (i + 1) < seq_length ? nucs[i + 1] : -1;
                        int nucjnext = nucs[jnext];
                        int nucjnext_1 = (jnext - 1) > -1 ? nucs[jnext - 1] : -1;

                        // 1. extend h(i, j) to h(i, jnext)
                        // double newscore;

                        double newscore = score_hairpin(i, jnext, nuci, nuci1, nucjnext_1, nucjnext);
                        // this candidate must be the best one at [i, jnext]
                        // so no need to check the score
                        update_if_better(bestH[jnext][i], newscore, MANNER_H);
                        ++nos_H;
                    }
                }
            }
        }
        if (j == 0) continue;

        // beam_size of Multi
        {
            if (beam_size > 0 && beamstepMulti.size() > beam_size) beam_prune(beamstepMulti);

            // for every state in Multi[j]
            //   1. extend (i, j) to (i, jnext)
            //   2. generate P (i, j)
            for(auto& item : beamstepMulti) {
                int i = item.first;
                LinearFoldState<double>& state = item.second;
                int nuci = nucs[i];
                int nuci1 = nucs[i+1];
                int jnext = next_pair[nuci][j];

                // 2. generate P (i, j)
                // lisiz, change the order because of the constraits
                {
                    double newscore = state.score + score_multi(i, j, nuci, nuci1, nucs[j-1], nucj, seq_length);
                    update_if_better(beamstepP[i], newscore, MANNER_P_EQ_MULTI);
                    ++ nos_P;
                }

                // lisiz cnstriants
                if (jnext != -1 && use_constraints) {
                    int nucjnext = nucs[jnext];
                    if (jnext > allow_unpaired_range[j] || !allow_paired(i, jnext, cons, nuci, nucjnext))
                        continue;
                }

                // 1. extend (i, j) to (i, jnext)
                {
                    char new_l1 = state.trace.paddings.l1;
                    int new_l2 = state.trace.paddings.l2 + jnext - j;
                    // if (jnext != -1 && new_l1 + new_l2 <= SINGLE_MAX_LEN) {
                    if (jnext != -1) {
                        // 1. extend (i, j) to (i, jnext)
                        double newscore = state.score + score_multi_unpaired(j, jnext - 1);
                        // this candidate must be the best one at [i, jnext]
                        // so no need to check the score
                        update_if_better(bestMulti[jnext][i], newscore, MANNER_MULTI_EQ_MULTI_PLUS_U,
                                         new_l1,
                                         new_l2
                        );
                        ++nos_Multi;
                    }
                }
            }
        }

        // beam_size of P
        {
            if (beam_size > 0 && beamstepP.size() > beam_size) beam_prune(beamstepP);

            // for every state in P[j]
            //   1. generate new helix/bulge
            //   2. M = P
            //   3. M2 = M + P
            //   4. C = C + P
            bool use_cube_pruning = beam_size > MIN_CUBE_PRUNING_SIZE
                                    && beamstepP.size() > MIN_CUBE_PRUNING_SIZE;            

            for(auto& item : beamstepP) {
                int i = item.first;
                LinearFoldState<double>& state = item.second;
                int nuci = nucs[i];
                int nuci_1 = (i-1>-1) ? nucs[i-1] : -1;

                // 2. M = P
                if(i > 0 && j < seq_length-1) {
                    double newscore = score_M1(i, j, j, nuci_1, nuci, nucj, nucj1, seq_length) + state.score;
                    update_if_better(beamstepM[i], newscore, MANNER_M_EQ_P);
                    ++ nos_M;
                }
                //printf(" M = P at %d\n", j); fflush(stdout);

                // 3. M2 = M + P
                if(!use_cube_pruning) {
                    int k = i - 1;
                    if ( k > 0 && !bestM[k].empty()) {
                        double M1_score = score_M1(i, j, j, nuci_1, nuci, nucj, nucj1, seq_length) + state.score;
                        // candidate list
                        auto bestM2_iter = beamstepM2.find(i);
                        if (bestM2_iter==beamstepM2.end() || M1_score > bestM2_iter->second.score) {
                            for (auto &m : bestM[k]) {
                                int newi = m.first;
                                // eq. to first convert P to M1, then M2/M = M + M1
                                double newscore = M1_score + m.second.score;
                                update_if_better(beamstepM2[newi], newscore, MANNER_M2_EQ_M_PLUS_P, k);
                                //update_if_better(bestM[j][newi], newscore, MANNER_M_eq_M_plus_P, k);
                                ++nos_M2;
                                //++nos_M;
                            }
                        }
                    }
                }
                //printf(" M/M2 = M + P at %d\n", j); fflush(stdout);

                // 4. C = C + P
                {
                    int k = i - 1;
                    if (k >= 0) {
                      LinearFoldState<double>& prefix_C = bestC[k];
                      if (prefix_C.manner != MANNER_NONE) {
                        int nuck = nuci_1;
                        int nuck1 = nuci;
                        double  newscore = score_external_paired(k+1, j, nuck, nuck1,
                                                             nucj, nucj1, seq_length) +
                                prefix_C.score + state.score;

                        update_if_better(beamstepC, newscore, MANNER_C_EQ_C_PLUS_P, k);
                        ++ nos_C;
                      }
                    } else {
                        double newscore = score_external_paired(0, j, -1, nucs[0],
                                                             nucj, nucj1, seq_length) +
                                state.score;
                        update_if_better(beamstepC, newscore, MANNER_C_EQ_C_PLUS_P, -1);
                        ++ nos_C;
                    }
                }
                //printf(" C = C + P at %d\n", j); fflush(stdout);

                // 1. generate new helix / single_branch
                // new state is of shape p..i..j..q
                if (i >0 && j<seq_length-1) {
                    double precomputed = score_junction_B(j, i, nucj, nucj1, nuci_1, nuci);
                    for (int p = i - 1; p >= std::max(i - SINGLE_MAX_LEN, 0); --p) {
                        int nucp = nucs[p];
                        int nucp1 = nucs[p + 1]; // hzhang: move here
                        int q = next_pair[nucp][j];

                        // lisiz constraints
                        if (use_constraints) {
                            if (p < i-1 && !allow_unpaired_position[p+1]) // lisiz: if p+1 must be paired, break
                                break;
                            if (!allow_unpaired_position[p]) {             // lisiz: if p must be paired, p must be left bracket
                                q = cons[p];
                                if (q < p) break;
                            }
                        }

                        while (q != -1 && ((i - p) + (q - j) - 2 <= SINGLE_MAX_LEN)) {
                            int nucq = nucs[q];

                            // lisiz constraints
                            if (use_constraints) {
                                if (q>j+1 && q > allow_unpaired_range[j])  // lisiz: if q-1 must be paired, break
                                    break;
                                if (!allow_paired(p, q, cons, nucp, nucq)) // lisiz: if p q are )(, break
                                    break;
                            }

                            int nucq_1 = nucs[q - 1];

                            if (p == i - 1 && q == j + 1) {
                                // helix
                                double newscore = score_helix(nucp, nucp1, nucq_1, nucq) + state.score;
                                update_if_better(bestP[q][p], newscore, MANNER_HELIX);
                                ++nos_P;
                            } else {
                                // single branch
                                double newscore = score_junction_B(p, q, nucp, nucp1, nucq_1, nucq) +
                                        precomputed +
                                        score_single_without_junctionB(p, q, i, j,
                                                                       nuci_1, nuci, nucj, nucj1) +
                                        state.score;
                                update_if_better(bestP[q][p], newscore, MANNER_SINGLE,
                                                 static_cast<char>(i - p),
                                                 q - j);
                                ++nos_P;
                            }
                            q = next_pair[nucp][q];
                        }
                    }
                }
                //printf(" helix / single at %d\n", j); fflush(stdout);
            }

            if (use_cube_pruning) {
                // 3. M2 = M + P with cube pruning
                vector<int> valid_Ps;
                vector<double> M1_scores;

                for(auto& item : beamstepP) {
                    int i = item.first;
                    LinearFoldState<double>& state = item.second;
                    int nuci = nucs[i];
                    int nuci_1 = (i - 1 > -1) ? nucs[i - 1] : -1;
                    int k = i - 1;

                    // group candidate Ps
                    if (k > 0 && !bestM[k].empty()) {
                        assert(bestM[k].size() == sorted_bestM[k].size());
                        double M1_score = score_M1(i, j, j, nuci_1, nuci, nucj, nucj1, seq_length) + state.score;
                        auto bestM2_iter = beamstepM2.find(i);
                        if (bestM2_iter == beamstepM2.end() || M1_score > bestM2_iter->second.score) {
                            valid_Ps.push_back(i);
                            M1_scores.push_back(M1_score);
                        }

                    }
                }

                // build max heap
                // heap is of form (heuristic score, (index of i in valid_Ps, index of M in bestM[i-1]))
                vector<pair<double, pair<int, int>>> heap;
                for (int p = 0; p < valid_Ps.size(); ++p) {
                    int i = valid_Ps[p];
                    int k = i - 1;
                    heap.push_back(make_pair(M1_scores[p] + sorted_bestM[k][0].first,
                                             make_pair(p, 0)
                    ));
                    push_heap(heap.begin(), heap.end());
                }

                // start cube pruning
                // stop after beam_size size M2 states being filled
                int filled = 0;
                // exit when filled >= beam_size and current score < prev score
                double prev_score = value_min;
                double current_score = value_min;
                // while ((filled < beam_size || current_score == prev_score) && !heap.empty()) {
                while ((filled < beam_size || (abs(current_score - prev_score) < 1e-7)) && !heap.empty()) {
                    auto &top = heap.front();
                    prev_score = current_score;
                    current_score = top.first;
                    int index_P = top.second.first;
                    int index_M = top.second.second;
                    int i = valid_Ps[top.second.first];
                    int k = i - 1;
                    int newi = sorted_bestM[k][index_M].second;
                    double newscore = M1_scores[index_P] + bestM[k][newi].score;
                    pop_heap(heap.begin(), heap.end());
                    heap.pop_back();

                    if (beamstepM2[newi].manner == MANNER_NONE) {
                        ++filled;
                        update_if_better(beamstepM2[newi], newscore, MANNER_M2_EQ_M_PLUS_P, k);
                        ++nos_M2;
                    } else {
                        assert(beamstepM2[newi].score > newscore - 1e-8);
                    }

                    ++index_M;
                    while (index_M < sorted_bestM[k].size()) {
                        // candidate_score is a heuristic score
                        double candidate_score = M1_scores[index_P] + sorted_bestM[k][index_M].first;
                        int candidate_newi = sorted_bestM[k][index_M].second;
                        if (beamstepM2.find(candidate_newi) == beamstepM2.end()) {
                            heap.push_back(make_pair(candidate_score,
                                                     make_pair(index_P, index_M)));
                            push_heap(heap.begin(), heap.end());
                            break;
                        } else {
                            // based on the property of cube pruning, the new score must be worse
                            // than the state already inserted
                            // so we keep iterate through the candidate list to find the next
                            // candidate
                            ++index_M;
                            assert(beamstepM2[candidate_newi].score >
                                   M1_scores[index_P] + bestM[k][candidate_newi].score - 1e-8);
                        }
                    }
                }
            }
        }
        //printf("P at %d\n", j); fflush(stdout);

        // beam_size of M2
        {
            if (beam_size > 0 && beamstepM2.size() > beam_size) beam_prune(beamstepM2);

            // for every state in M2[j]
            //   1. multi-loop  (by extending M2 on the left)
            //   2. M = M2
            for(auto& item : beamstepM2) {
                int i = item.first;
                LinearFoldState<double>& state = item.second;

                // 2. M = M2
                {
                    update_if_better(beamstepM[i], state.score, MANNER_M_EQ_M2);
                    ++ nos_M;
                }

                // 1. multi-loop
                {
                    for (int p = i-1; p >= std::max(i - SINGLE_MAX_LEN, 0); --p) {
                        int nucp = nucs[p];
                        int q = next_pair[nucp][j];

                        if (use_constraints) {
                            if (p < i - 1 && !allow_unpaired_position[p+1])
                                break;
                            if (!allow_unpaired_position[p]) {
                                q = cons[p];
                                if (q < p) break;
                            }
                            if (q > j+1 && q > allow_unpaired_range[j])
                                continue;
                            int nucq = nucs[q];
                            if (!allow_paired(p, q, cons, nucp, nucq))
                                continue;
                        }

                        if (q != -1 && ((i - p - 1) <= SINGLE_MAX_LEN)) {
                            // the current shape is p..i M2 j ..q
                            double newscore = score_multi_unpaired(p+1, i-1) +
                                    score_multi_unpaired(j+1, q-1) + state.score;
                            update_if_better(bestMulti[q][p], newscore, MANNER_MULTI,
                                             static_cast<char>(i - p),
                                             q - j);
                            ++ nos_Multi;

                            //q = next_pair[nucp][q];
                        }
                    }
                }
            }
        }
        //printf("M2 at %d\n", j); fflush(stdout);

        // beam_size of M
        {
            double threshold = value_min;
            if (beam_size > 0 && beamstepM.size() > beam_size) threshold = beam_prune(beamstepM);

            sort_m(threshold, beamstepM, sorted_bestM[j]);

            // for every state in M[j]
            //   1. M = M + unpaired
            for(auto& item : beamstepM) {
                int i = item.first;
                LinearFoldState<double>& state = item.second;
                if (j < seq_length-1) {
                    if (use_constraints && !allow_unpaired_position[j+1]) // if j+1 must be paired
                        continue;
                    double newscore = score_multi_unpaired(j + 1, j + 1) + state.score;
                    update_if_better(bestM[j+1][i], newscore, MANNER_M_EQ_M_PLUS_U);
                    ++ nos_M;
                }
            }
        }
        //printf("M at %d\n", j); fflush(stdout);

        // beam_size of C
        {
            // C = C + U
            if (j < seq_length-1) {
                if (use_constraints && !allow_unpaired_position[j+1])
                        continue;

                double newscore = score_external_unpaired(j+1, j+1) + beamstepC.score;

                update_if_better(bestC[j+1], newscore, MANNER_C_EQ_C_PLUS_U);
                ++ nos_C;
            }
        }
        //printf("C at %d\n", j); fflush(stdout);

    }  // end of for-loo j

    LinearFoldState<double>& viterbi = bestC[seq_length-1];

    char result[seq_length + 1];
    if (!get_parentheses(result, seq)) {
        return decoder_result;
    }

    // post_process
    next_pair->clear();

    unsigned long nos_tot = nos_H + nos_P + nos_M2 + nos_Multi + nos_M + nos_C;

    double printscore = viterbi.score;

    decoder_result.structure = string(result);
    decoder_result.score = viterbi.score;

    // post_process();

    return decoder_result;
}

