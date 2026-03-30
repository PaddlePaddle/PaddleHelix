"""
analyze ensemble results
"""
import sys
import os
import numpy as np
import glob
import pandas as pd


def get_ensemble_result(ensem_list, score_name, en_type='', conf_name=None):
    if en_type == 'mean':
        en_scores = [l[score_name] for l in ensem_list if score_name in l]
        if len(en_scores) == 0:
            return None
        final_score = np.array([np.mean(en_scores)])
    
    elif en_type == 'max':
        en_scores = [l[score_name] for l in ensem_list if score_name in l]
        if len(en_scores) == 0:
            return None
        final_score = np.array([np.max(en_scores[:i]) for i in range(1, len(en_scores) + 1)])
    
    elif en_type == 'min':
        en_scores = [l[score_name] for l in ensem_list if score_name in l]
        if len(en_scores) == 0:
            return None
        final_score = np.array([np.min(en_scores[:i]) for i in range(1, len(en_scores) + 1)])
    
    elif en_type == 'conf':
        if not conf_name in ensem_list[0]:
            return None
        mat = np.array([[l[conf_name], l[score_name]] for l in ensem_list
                if score_name in l and conf_name in l])
        if len(mat) == 0:
            return None
        final_score = []
        for i in range(1, len(mat) + 1):
            conf_values = mat[:i, 0]
            en_scores = mat[:i, 1]
            score = en_scores[np.argmax(conf_values)]
            final_score.append(score)
        final_score = np.array(final_score)
    
    else:
        raise ValueError(f'Unknown type {en_type}')
    
    return final_score


def generate_ensemble_report(all_name_list, all_scores_list, 
        target_score_name, confidence_names=None, out_file=None, shuffle_tries=True,
        gen_case_wise_report=False, output_median=False):
    """
    all_name_list: list of sample names. repeated names are for the same protein.
    all_scores_list: list of scores. each score is a dict of {name: value}
    """
    if confidence_names is None:
        confidence_names = ['mean_plddt', 'ptm', 'iptm', 
                'ranking_confidence', 'chain_pair_iptm-ligand',
                'pred_interface_neg_avgPAE', 'actifpTM', 'actifpTM_interfaceMask',
                'ligand_iptm', 'ligand_iptm_mean', 'ligand_mean_plddt']
    
    args_list = [
        {'en_type': 'mean'},
        {'en_type': 'max'},
        {'en_type': 'min'}]
    for conf_name in confidence_names:
        if conf_name in all_scores_list[0]:
            args_list.append({'en_type': 'conf', 'conf_name': conf_name})
    
    if not out_file is None:
        out_f = open(out_file, 'w')

    out_dir = os.path.dirname(out_file)
    ensemble_cases_dir = os.path.join(out_dir, 'ensemble_cases')
    os.makedirs(ensemble_cases_dir, exist_ok=True)

    uniq_prot_names = np.unique(all_name_list)
    if gen_case_wise_report:
        case_out_f = open(out_file + '.case', 'w')

    if shuffle_tries:
        try_out = 8
    else:
        try_out = 1
    ## go over all confidence_names
    print('Generating ensemble report:')
    for args in args_list:
        raw_final_score_mat = []    # (n_prot, n_ensemble)
        ## go over all proteins
        for prot in uniq_prot_names:
            ensem_list = [x for x, y in zip(all_scores_list, all_name_list) if prot == y]
            try_out_list = []
            for _ in range(try_out):
                if shuffle_tries:
                    np.random.shuffle(ensem_list)
                final_score = get_ensemble_result(ensem_list, target_score_name, **args)
                try_out_list.append(final_score)
            if try_out_list[0] is None:
                continue
            mean_final_score = np.stack(try_out_list).mean(0)
            raw_final_score_mat.append(mean_final_score)

        if gen_case_wise_report:
            str_args = f'{args}'.replace(' ', '')
            str_title = f'\n>>>>>>>>>>> {target_score_name}\t{str_args}'
            print(str_title, file=case_out_f)

            for i, final_score in enumerate(raw_final_score_mat):
                prot = uniq_prot_names[i]
                num_ensem = len(final_score)
                str_score = ' '.join([f'{x:.3f}' for x in final_score])
                out_str = f'{prot}\tnum_ensem:{num_ensem}\t{str_score}'
                print(out_str, file=case_out_f)

        num_ensem = len(raw_final_score_mat[0])
        final_score_mat = [x for x in raw_final_score_mat if len(x) == num_ensem]
        final_score_mat = np.array(final_score_mat)
        if output_median:
            avg_scores = np.median(final_score_mat, axis=0)
        else:
            avg_scores = final_score_mat.mean(0)

        str_args = f'{args}'.replace(' ', '')
        num_prot = len(final_score_mat)
        str_score = ' '.join([f'{x:.3f}' for x in avg_scores])
        out_str = f'{target_score_name}\t{str_args}\tnum_prot:{num_prot}\tnum_ensem:{num_ensem}\t{str_score}'
        print(out_str)
        if not out_file is None:
            out_f.write(out_str + '\n')

        # Create an empty DataFrame
        df = pd.DataFrame()

        # Iterate over final_score_mat and uniq_prot_names to populate data into the DataFrame
        for i, final_score in enumerate(final_score_mat):
            prot_name = uniq_prot_names[i]
            # Assuming final_score is a one-dimensional array, we need to convert it to a list or Series to add to the DataFrame
            score_list = final_score.tolist() if isinstance(final_score, np.ndarray) else final_score
            # Create a temporary DataFrame
            temp_df = pd.DataFrame([score_list], columns=[f'{j}' for j in range(len(score_list))])
            # Insert protein name into the first column
            temp_df.insert(0, 'prot_name', prot_name)
            # Append to the main DataFrame
            df = pd.concat([df, temp_df], ignore_index=True)

        case_score_file = os.path.join(ensemble_cases_dir, f'{target_score_name}_{str_args}.csv')
        df.to_csv(case_score_file, index=False)

    if not out_file is None:
        out_f.close()

    if gen_case_wise_report:
        case_out_f.close()
