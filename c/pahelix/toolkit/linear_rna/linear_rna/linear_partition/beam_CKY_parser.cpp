#include "linear_partition/beam_CKY_parser.h"

#include <stdio.h> 

#include "utils/utility.h"
#include "utils/utility_v.h"
#include "utils/quick_select.h"

using namespace std;

LinearPartitionBeamCKYParser::LinearPartitionBeamCKYParser(int beam_size,
                        char energy_model,
                        bool no_sharp_turn,
                        float bpp_cutoff) :
        beam_size(beam_size),
        energy_model(energy_model),
        no_sharp_turn(no_sharp_turn),
        bpp_cutoff(bpp_cutoff) {}

float LinearPartitionBeamCKYParser::parse(string& seq) {
    seq_length = static_cast<unsigned>(seq.length());
    prepare();
    for (int i = 0; i < seq_length; ++i) {
        nucs[i] = GET_ACGU_NUM(seq[i]);
    }
    vector<int> next_pair[NOTON];
    for (int nuci = 0; nuci < NOTON; ++nuci) {
        // next_pair
        next_pair[nuci].resize(seq_length, -1);
        int next = -1;
        for (int j = seq_length - 1; j >=0; --j) {
            next_pair[nuci][j] = next;
            if (g_allowed_pairs[nuci][nucs[j]]) {
                next = j;
            }
        }
    }

    if (energy_model == 'v') {
        v_init_tetra_hex_tri(seq, seq_length, if_tetraloops, if_hexaloops, if_triloops);
    }

    if (energy_model == 'v') {
        if (seq_length > 0) {
            bestC[0].alpha = 0.0;
        } 
        if (seq_length > 1) {
            bestC[1].alpha = 0.0;
        }
    } else {
        if (seq_length > 0) {
            fast_log_plus_equals(bestC[0].alpha, score_external_unpaired(0, 0));
        } 
        if (seq_length > 1) {
            fast_log_plus_equals(bestC[1].alpha, score_external_unpaired(0, 1));
        }
    }

    for (int j = 0; j < seq_length; ++j) {
        int nucj = nucs[j];
        int nucj1 = (j + 1) < seq_length ? nucs[j+1] : -1;

        unordered_map<int, LinearPartitionState>& beamstepH = bestH[j];
        unordered_map<int, LinearPartitionState>& beamstepMulti = bestMulti[j];
        unordered_map<int, LinearPartitionState>& beamstepP = bestP[j];
        unordered_map<int, LinearPartitionState>& beamstepM2 = bestM2[j];
        unordered_map<int, LinearPartitionState>& beamstepM = bestM[j];
        LinearPartitionState& beamstepC = bestC[j];

        // beam_sizeof H
        {
            if (beam_size> 0 && beamstepH.size() > beam_size) {
                beam_prune(beamstepH);
            }

            {
                // for nucj put H(j, j_next) into H[j_next]
                int jnext = next_pair[nucj][j];
                if (no_sharp_turn) {
                    while (jnext - j < 4 && jnext != -1) {
                        jnext = next_pair[nucj][jnext];
                    }
                } 
                if (jnext != -1) {
                    int nucjnext = nucs[jnext];
                    int nucjnext_1 = (jnext - 1) > -1 ? nucs[jnext - 1] : -1;

                    if (energy_model == 'v') {
                        int tetra_hex_tri = -1;
                        if (jnext - j - 1 == 4) {// 6:tetra
                            tetra_hex_tri = if_tetraloops[j];
                        } else if (jnext - j - 1 == 6) { // 8:hexa
                            tetra_hex_tri = if_hexaloops[j];
                        } else if (jnext - j - 1 == 3) {// 5:tri
                            tetra_hex_tri = if_triloops[j];
                        }

                        float newscore = - v_score_hairpin(j, jnext, nucj, nucj1, nucjnext_1, nucjnext, tetra_hex_tri);
                        fast_log_plus_equals(bestH[jnext][j].alpha, newscore / kT);
                    }
                    else {
                        float newscore = score_hairpin(j, jnext, nucj, nucj1, nucjnext_1, nucjnext);
                        fast_log_plus_equals(bestH[jnext][j].alpha, newscore);
                    }
                }
            }

            {
                // for every state h in H[j]
                //   1. extend h(i, j) to h(i, jnext)
                //   2. generate p(i, j)
                for (auto &item : beamstepH) {
                    int i = item.first;
                    LinearPartitionState &state = item.second;
                    int nuci = nucs[i];
                    int jnext = next_pair[nuci][j];

                    if (jnext != -1) {
                        int nuci1 = (i + 1) < seq_length ? nucs[i + 1] : -1;
                        int nucjnext = nucs[jnext];
                        int nucjnext_1 = (jnext - 1) > -1 ? nucs[jnext - 1] : -1;

                        // 1. extend h(i, j) to h(i, jnext)
                        if (energy_model == 'v') {
                            int tetra_hex_tri = -1;
                            if (jnext-i-1 == 4) { // 6:tetra
                                tetra_hex_tri = if_tetraloops[i];
                            } else if (jnext-i-1 == 6) { // 8:hexa
                                tetra_hex_tri = if_hexaloops[i];
                            } else if (jnext-i-1 == 3) { // 5:tri
                                tetra_hex_tri = if_triloops[i];
                            }
                            float newscore = - v_score_hairpin(i, jnext, nuci, nuci1, nucjnext_1, nucjnext, tetra_hex_tri);
                            fast_log_plus_equals(bestH[jnext][i].alpha, (newscore/kT));
                        } else {
                            float newscore = score_hairpin(i, jnext, nuci, nuci1, nucjnext_1, nucjnext);
                            fast_log_plus_equals(bestH[jnext][i].alpha, newscore);
                        }
                    }

                    // 2. generate p(i, j)
                    fast_log_plus_equals(beamstepP[i].alpha, state.alpha);
                }
            }
        }
        if (j == 0) continue;

        // beam_sizeof Multi
        {
            if (beam_size> 0 && beamstepMulti.size() > beam_size) {
                beam_prune(beamstepMulti);
            }

            for(auto& item : beamstepMulti) {
                int i = item.first;
                LinearPartitionState& state = item.second;

                int nuci = nucs[i];
                int nuci1 = nucs[i+1];
                int jnext = next_pair[nuci][j];

                // 1. extend (i, j) to (i, jnext)
                {
                    if (jnext != -1) {
                        if (energy_model == 'v') {
                            fast_log_plus_equals(bestMulti[jnext][i].alpha, (state.alpha));
                        } else {   
                            float newscore = score_multi_unpaired(j, jnext - 1);
                            fast_log_plus_equals(bestMulti[jnext][i].alpha, state.alpha + newscore);
                        }                                        
                    }
                }

                // 2. generate P (i, j)
                {
                    if (energy_model == 'v') {
                        int score_multi = - v_score_multi(i, j, nuci, nuci1, nucs[j - 1], nucj, seq_length);
                        fast_log_plus_equals(beamstepP[i].alpha, (state.alpha + score_multi/kT));
                    }
                    else{
                        float newscore = score_multi(i, j, nuci, nuci1, nucs[j - 1], nucj, seq_length);
                        fast_log_plus_equals(beamstepP[i].alpha, state.alpha + newscore);
                    }
                }
            }
        }

        // beam_sizeof P
        {   
            if (beam_size> 0 && beamstepP.size() > beam_size) {
                beam_prune(beamstepP);
            }

            // for every state in P[j]
            //   1. generate new helix/bulge
            //   2. M = P
            //   3. M2 = M + P
            //   4. C = C + P
            for(auto& item : beamstepP) {
                int i = item.first;
                LinearPartitionState& state = item.second;
                int nuci = nucs[i];
                int nuci_1 = (i - 1 > -1) ? nucs[i - 1] : -1;

                // 1. generate new helix / single_branch
                // new state is of shape p..i..j..q
                if (i > 0 && j < seq_length - 1) {      
                    float precomputed = 0;
                    if (energy_model == 'c') {
                        precomputed = score_junction_B(j, i, nucj, nucj1, nuci_1, nuci);
                    }             
                    for (int p = i - 1; p >= std::max(i - SINGLE_MAX_LEN, 0); --p) {
                        int nucp = nucs[p];
                        int nucp1 = nucs[p + 1]; 
                        int q = next_pair[nucp][j];
                        while (q != -1 && ((i - p) + (q - j) - 2 <= SINGLE_MAX_LEN)) {
                            int nucq = nucs[q];
                            int nucq_1 = nucs[q - 1];

                            if (p == i - 1 && q == j + 1) {
                                // helix                  
                                if (energy_model == 'v') {
                                    int score_single = -v_score_single(p,q,i,j, nucp, nucp1, nucq_1, nucq,
                                                             nuci_1, nuci, nucj, nucj1);
                                    fast_log_plus_equals(bestP[q][p].alpha, (state.alpha + score_single/kT));
                                }
                                else{
                                    float newscore = score_helix(nucp, nucp1, nucq_1, nucq);
                                    fast_log_plus_equals(bestP[q][p].alpha, state.alpha + newscore);
                                }
                            } else {
                                // single branch                 
                                if (energy_model == 'v') {
                                    int score_single = - v_score_single(p,q,i,j, nucp, nucp1, nucq_1, nucq,
                                                   nuci_1, nuci, nucj, nucj1);
                                    fast_log_plus_equals(bestP[q][p].alpha, (state.alpha + score_single/kT));
                                }
                                else{
                                    float newscore = score_junction_B(p, q, nucp, nucp1, nucq_1, nucq) +
                                        precomputed +
                                        score_single_without_junctionB(p, q, i, j,
                                                                       nuci_1, nuci, nucj, nucj1);
                                    fast_log_plus_equals(bestP[q][p].alpha, state.alpha + newscore);
                                }
                            }
                            q = next_pair[nucp][q];
                        }
                    }
                }

                // 2. M = P
                if(i > 0 && j < seq_length-1) { 
                    if (energy_model == 'v') {
                        int score_M1 = - v_score_M1(i, j, j, nuci_1, nuci, nucj, nucj1, seq_length);
                        fast_log_plus_equals(beamstepM[i].alpha, (state.alpha + score_M1/kT));
                    }
                    else{
                        float newscore = score_M1(i, j, j, nuci_1, nuci, nucj, nucj1, seq_length);
                        fast_log_plus_equals(beamstepM[i].alpha, state.alpha + newscore);
                    }
                }

                // 3. M2 = M + P
                int k = i - 1;
                if ( k > 0 && !bestM[k].empty()) {
                    float m1_alpha = 0.0; 
                    if (energy_model == 'v') {
                        int M1_score = - v_score_M1(i, j, j, nuci_1, nuci, nucj, nucj1, seq_length);
                        m1_alpha = state.alpha + M1_score/kT;
                    } 
                    else{
                        float M1_score = score_M1(i, j, j, nuci_1, nuci, nucj, nucj1, seq_length);
                        m1_alpha = state.alpha + M1_score;
                    }
                    for (auto &m : bestM[k]) {
                        int newi = m.first;
                        LinearPartitionState& m_state = m.second;
                        fast_log_plus_equals(beamstepM2[newi].alpha, m_state.alpha + m1_alpha);
                    }
                }

                // 4. C = C + P
                {
                    int k = i - 1;
                    if (k >= 0) {
                      LinearPartitionState& prefix_C = bestC[k];
                        int nuck = nuci_1;
                        int nuck1 = nuci;              
                        if (energy_model == 'v') {
                            int score_external_paired = - v_score_external_paired(k + 1, j, nuck, nuck1,
                                                                nucj, nucj1, seq_length);
                            fast_log_plus_equals(beamstepC.alpha, prefix_C.alpha + state.alpha + score_external_paired/kT);
                        }
                        else{
                            float newscore = score_external_paired(k + 1, j, nuck, nuck1,
                                                                nucj, nucj1, seq_length);
                            fast_log_plus_equals(beamstepC.alpha, prefix_C.alpha + state.alpha + newscore);
                        }
                    } else {         
                        if (energy_model == 'v') {
                            int score_external_paired = - v_score_external_paired(0, j, -1, nucs[0],
                                                                    nucj, nucj1, seq_length);
                            fast_log_plus_equals(beamstepC.alpha, state.alpha + score_external_paired/kT);  
                        }     
                        else {
                            float newscore = score_external_paired(0, j, -1, nucs[0],
                                                                nucj, nucj1, seq_length);
                            fast_log_plus_equals(beamstepC.alpha, state.alpha + newscore);
                        }
                    }
                }
            }
        }


        // beam_sizeof M2
        {
            if (beam_size> 0 && beamstepM2.size() > beam_size) {
                beam_prune(beamstepM2);
            }

            for(auto& item : beamstepM2) {
                int i = item.first;
                LinearPartitionState& state = item.second;

                // 1. multi-loop
                {
                    for (int p = i-1; p >= std::max(i - SINGLE_MAX_LEN, 0); --p) {
                        int nucp = nucs[p];
                        int q = next_pair[nucp][j];
                        if (q != -1 && ((i - p - 1) <= SINGLE_MAX_LEN)) {
                            if (energy_model == 'v') {
                                fast_log_plus_equals(bestMulti[q][p].alpha, state.alpha);
                            }
                            else{
                                float newscore = score_multi_unpaired(p+1, i-1) +
                                        score_multi_unpaired(j+1, q-1);
                                fast_log_plus_equals(bestMulti[q][p].alpha, state.alpha + newscore);      
                            }            
                        }
                    }
                }

                // 2. M = M2
                fast_log_plus_equals(beamstepM[i].alpha, state.alpha);  
            }
        }

        // beam_sizeof M
        {
            // float threshold = VALUE_MIN;
            if (beam_size> 0 && beamstepM.size() > beam_size) {
                beam_prune(beamstepM);
            }

            for(auto& item : beamstepM) {
                int i = item.first;
                LinearPartitionState& state = item.second;
                if (j < seq_length-1) {
                    if (energy_model == 'v') {
                        fast_log_plus_equals(bestM[j+1][i].alpha, state.alpha); 
                    }
                    else{
                        float newscore = score_multi_unpaired(j + 1, j + 1);
                        fast_log_plus_equals(bestM[j+1][i].alpha, state.alpha + newscore); 
                    }
                }
            }
        }

        // beam_sizeof C
        {
            // C = C + U
            if (j < seq_length-1) {
                if (energy_model == 'v') {
                    fast_log_plus_equals(bestC[j+1].alpha, beamstepC.alpha); 
                }   
                else{
                    float newscore = score_external_unpaired(j+1, j+1);
                    fast_log_plus_equals(bestC[j+1].alpha, beamstepC.alpha + newscore); 
                }
            }
        }

    }  // end of for-loo j

    LinearPartitionState& viterbi = bestC[seq_length-1];

    // unsigned long nos_tot = nos_H + nos_P + nos_M2 + nos_Multi + nos_M + nos_C;

    outside(next_pair);
    cal_pair_probs(viterbi);

    // post_process();

    if (energy_model == 'v')
        return float(-kT * viterbi.alpha / 100.0);
    else 
        return viterbi.alpha;
}

void LinearPartitionBeamCKYParser::prepare() {
    nucs = new int[seq_length];
    bestC = new LinearPartitionState[seq_length];
    bestH = new unordered_map<int, LinearPartitionState>[seq_length];
    bestP = new unordered_map<int, LinearPartitionState>[seq_length];
    bestM = new unordered_map<int, LinearPartitionState>[seq_length];
    bestM2 = new unordered_map<int, LinearPartitionState>[seq_length];
    bestMulti = new unordered_map<int, LinearPartitionState>[seq_length];
    
    scores.clear();
    scores.reserve(seq_length);

    pij.clear();
    if_tetraloops.clear();
    if_hexaloops.clear();
    if_triloops.clear();
}

void LinearPartitionBeamCKYParser::post_process() {
    delete[] bestC;  
    delete[] bestH;  
    delete[] bestP;  
    delete[] bestM;  
    delete[] bestM2;  
    delete[] bestMulti;  

    delete[] nucs;
    scores.clear();
    pij.clear();
    if_tetraloops.clear();
    if_hexaloops.clear();
    if_triloops.clear();
}

void LinearPartitionBeamCKYParser::cal_pair_probs(LinearPartitionState& viterbi) {
    for(int j=0; j<seq_length; j++) {
        for(auto &item : bestP[j]) {
            int i = item.first;
            LinearPartitionState state = item.second;
            
            float temp_prob_inside = state.alpha + state.beta - viterbi.alpha;
            if (temp_prob_inside > float(-9.91152)) {
                float prob = fast_exp(temp_prob_inside);
                if(prob > 1.0) prob = 1.0;
                if(prob < bpp_cutoff) continue;
                pij[make_pair(i+1, j+1)] = prob;
            }
        }
    }
    return;
}

void LinearPartitionBeamCKYParser::outside(vector<int> next_pair[]) {
    bestC[seq_length-1].beta = 0.0;

    // from right to left
    // value_type newscore = 0.0;
    for(int j = seq_length-1; j > 0; --j) {
        int nucj = nucs[j];
        int nucj1 = (j+1) < seq_length ? nucs[j+1] : -1;

        unordered_map<int, LinearPartitionState>& beamstepH = bestH[j];
        unordered_map<int, LinearPartitionState>& beamstepMulti = bestMulti[j];
        unordered_map<int, LinearPartitionState>& beamstepP = bestP[j];
        unordered_map<int, LinearPartitionState>& beamstepM2 = bestM2[j];
        unordered_map<int, LinearPartitionState>& beamstepM = bestM[j];
        LinearPartitionState& beamstepC = bestC[j];

        // beam_sizeof C
        {
            // C = C + U
            if (j < seq_length-1) {
                if (energy_model == 'v') {
                    fast_log_plus_equals(beamstepC.beta, (bestC[j+1].beta));
                }              
                else{
                    float newscore = score_external_unpaired(j+1, j+1);
                    fast_log_plus_equals(beamstepC.beta, bestC[j+1].beta + newscore);
                }
            }
        }
    
        // beam_sizeof M
        {
            for(auto& item : beamstepM) {
                int i = item.first;
                LinearPartitionState& state = item.second;
                if (j < seq_length-1) {
                    if (energy_model == 'v') {
                        fast_log_plus_equals(state.beta, bestM[j+1][i].beta);
                    }
                    else {
                        float newscore = score_multi_unpaired(j + 1, j + 1);
                        fast_log_plus_equals(state.beta, bestM[j+1][i].beta + newscore);
                    }                 
                }
            }
        }

        // beam_sizeof M2
        {
            for(auto& item : beamstepM2) {
                int i = item.first;
                LinearPartitionState& state = item.second;

                // 1. multi-loop
                {
                    for (int p = i-1; p >= std::max(i - SINGLE_MAX_LEN, 0); --p) {
                        int nucp = nucs[p];
                        int q = next_pair[nucp][j];
                        if (q != -1 && ((i - p - 1) <= SINGLE_MAX_LEN)) {
                            if (energy_model == 'v') {
                                fast_log_plus_equals(state.beta, bestMulti[q][p].beta);
                            }
                            else {
                                float newscore = score_multi_unpaired(p+1, i-1) +
                                    score_multi_unpaired(j+1, q-1);
                                fast_log_plus_equals(state.beta, bestMulti[q][p].beta + newscore);
                            }                      
                        }
                    }
                }

                // 2. M = M2
                fast_log_plus_equals(state.beta, beamstepM[i].beta);
            }
        }

        // beam_sizeof P
        {  
            for(auto& item : beamstepP) {
                int i = item.first;
                LinearPartitionState& state = item.second;
                int nuci = nucs[i];
                int nuci_1 = (i-1>-1) ? nucs[i-1] : -1;

                if (i >0 && j<seq_length-1) {  
                    float precomputed = 0;
                    if (energy_model == 'c') {
                        precomputed = score_junction_B(j, i, nucj, nucj1, nuci_1, nuci);
                    }
                    for (int p = i - 1; p >= std::max(i - SINGLE_MAX_LEN, 0); --p) {
                        int nucp = nucs[p];
                        int nucp1 = nucs[p + 1]; 
                        int q = next_pair[nucp][j];
                        while (q != -1 && ((i - p) + (q - j) - 2 <= SINGLE_MAX_LEN)) {
                            int nucq = nucs[q];
                            int nucq_1 = nucs[q - 1];

                            if (p == i - 1 && q == j + 1) {
                                // helix 
                                if (energy_model == 'v') {
                                    int score_single = -v_score_single(p,q,i,j, nucp, nucp1, nucq_1, nucq,
                                                                nuci_1, nuci, nucj, nucj1);
                                    fast_log_plus_equals(state.beta, (bestP[q][p].beta + score_single/kT));
                                }
                                else{
                                    float newscore = score_helix(nucp, nucp1, nucq_1, nucq);
                                    fast_log_plus_equals(state.beta, bestP[q][p].beta + newscore);
                                }                           
                            } else {
                                // single branch 
                                if (energy_model == 'v') {
                                    int score_single = - v_score_single(p,q,i,j, nucp, nucp1, nucq_1, nucq,
                                                    nuci_1, nuci, nucj, nucj1);
                                    fast_log_plus_equals(state.beta, (bestP[q][p].beta + score_single/kT));
                                }
                                else {
                                    float newscore = score_junction_B(p, q, nucp, nucp1, nucq_1, nucq) +
                                        precomputed + 
                                        score_single_without_junctionB(p, q, i, j, nuci_1, nuci, nucj, nucj1);
                                    fast_log_plus_equals(state.beta, bestP[q][p].beta + newscore);
                                }                            
                            }
                            q = next_pair[nucp][q];
                        }
                    }
                }

                // 2. M = P
                if(i > 0 && j < seq_length-1) {
                    if (energy_model == 'v') {
                        int score_M1 = - v_score_M1(i, j, j, nuci_1, nuci, nucj, nucj1, seq_length);
                        fast_log_plus_equals(state.beta, (beamstepM[i].beta + score_M1/kT));
                    }
                    else{
                        float newscore = score_M1(i, j, j, nuci_1, nuci, nucj, nucj1, seq_length);
                        fast_log_plus_equals(state.beta, beamstepM[i].beta + newscore);
                    }
                }

                // 3. M2 = M + P
                int k = i - 1;
                if ( k > 0 && !bestM[k].empty()) {
                    float m1_alpha = 0.0;
                    if (energy_model == 'v') {
                        int M1_score = - v_score_M1(i, j, j, nuci_1, nuci, nucj, nucj1, seq_length);
                        m1_alpha = M1_score/kT;
                    }
                    else {
                        float newscore = score_M1(i, j, j, nuci_1, nuci, nucj, nucj1, seq_length);
                        m1_alpha = newscore;
                    }
                    
                    float m1_plus_P_alpha = state.alpha + m1_alpha;
                    for (auto &m : bestM[k]) {
                        int newi = m.first;
                        LinearPartitionState& m_state = m.second;
                        fast_log_plus_equals(state.beta, (beamstepM2[newi].beta + m_state.alpha + m1_alpha));
                        fast_log_plus_equals(m_state.beta, (beamstepM2[newi].beta + m1_plus_P_alpha));
                    }
                }

                // 4. C = C + P
                {
                    int k = i - 1;
                    if (k >= 0) {
                        int nuck = nuci_1;
                        int nuck1 = nuci;
                        float external_paired_alpha_plus_beamstepC_beta = 0.0;
                        if (energy_model == 'v') {
                            int score_external_paired = - v_score_external_paired(k+1, j, nuck, nuck1,
                                                                    nucj, nucj1, seq_length);
                            external_paired_alpha_plus_beamstepC_beta = beamstepC.beta + score_external_paired/kT;
                        }
                        else{
                            float newscore = score_external_paired(k+1, j, nuck, nuck1, nucj, nucj1, seq_length);
                            external_paired_alpha_plus_beamstepC_beta = beamstepC.beta + newscore;
                        }                       
                        fast_log_plus_equals(bestC[k].beta, state.alpha + external_paired_alpha_plus_beamstepC_beta);
                        fast_log_plus_equals(state.beta, bestC[k].alpha + external_paired_alpha_plus_beamstepC_beta);
                    } else {
                        if (energy_model == 'v') {
                            int score_external_paired = - v_score_external_paired(0, j, -1, nucs[0],
                                                                    nucj, nucj1, seq_length);
                            fast_log_plus_equals(state.beta, (beamstepC.beta + score_external_paired/kT));
                        }
                        else {
                            float newscore = score_external_paired(0, j, -1, nucs[0],
                                                             nucj, nucj1, seq_length);
                            fast_log_plus_equals(state.beta, beamstepC.beta + newscore);
                        }                     
                    }
                }
            }
        }

        // beam_sizeof Multi
        {
            for(auto& item : beamstepMulti) {
                int i = item.first;
                LinearPartitionState& state = item.second;

                int nuci = nucs[i];
                int nuci1 = nucs[i+1];
                int jnext = next_pair[nuci][j];

                // 1. extend (i, j) to (i, jnext)
                {
                    if (jnext != -1) {
                        if (energy_model == 'v') {
                            fast_log_plus_equals(state.beta, (bestMulti[jnext][i].beta));
                        }
                        else {
                            float newscore = score_multi_unpaired(j, jnext - 1);
                            fast_log_plus_equals(state.beta, bestMulti[jnext][i].beta + newscore);
                        }
                    }
                }

                // 2. generate P (i, j)
                {
                    if (energy_model == 'v') {
                        int score_multi = - v_score_multi(i, j, nuci, nuci1, nucs[j-1], nucj, seq_length);
                        fast_log_plus_equals(state.beta, (beamstepP[i].beta + score_multi/kT));
                    }
                    else {
                        float newscore = score_multi(i, j, nuci, nuci1, nucs[j-1], nucj, seq_length);
                        fast_log_plus_equals(state.beta, beamstepP[i].beta + newscore);
                    }                 
                }
            }
        }
    }  // end of for-loo j


    return;
}

float LinearPartitionBeamCKYParser::beam_prune(std::unordered_map<int, LinearPartitionState> &beamstep) {
    scores.clear();
    for (auto &item : beamstep) {
        int i = item.first;
        LinearPartitionState &cand = item.second;
        int k = i - 1;
        float newalpha = (k >= 0 ? bestC[k].alpha : 0.0) + cand.alpha;
        scores.push_back(make_pair(newalpha, i));
    }
    if (scores.size() <= beam_size) {
        return VALUE_MIN;
    }
    float threshold = quickselect(scores, 0, scores.size() - 1, scores.size() - beam_size);
    for (auto &p : scores) {
        if (p.first < threshold) {
            beamstep.erase(p.second);
        }
    }

    return threshold;
}

