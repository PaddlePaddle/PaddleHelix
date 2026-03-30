import logging
import os
import signal
import tempfile
import time, gzip, pickle
import re
import math 
import json
import fcntl
from collections import OrderedDict
import numpy as np

from helixfold.data.tools import jackhmmer
from helixfold.common import residue_constants
from helixfold.data import parsers

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__file__)


def search_cache_exact(cache_dict_path, desc, seq, chain_id, feat_dir):
    """ load cache dict and perform exact match"""
    logger.info(f'[MSA/Template] {desc}; matching cache from {cache_dict_path}')
    if not os.path.exists(cache_dict_path):
        return None

    try:
        pkl_dir_hit = None
        with open(cache_dict_path, 'r') as cache_dict_file:
            seq_to_pkl = json.load(cache_dict_file)
            if seq in seq_to_pkl:
                pkl_dir_hit = seq_to_pkl[seq.replace("\n", "")]
        
        if not pkl_dir_hit is None:
            pkl_path = os.path.join(feat_dir, pkl_dir_hit, 'features.pkl.gz')
            if not os.path.exists(pkl_path):
                pkl_path = os.path.join(feat_dir, pkl_dir_hit, 'features.pkl')
            
            if not os.path.exists(pkl_path): 
                logger.info(f'[MSA/Template] {desc}; cannot find cached feature pkl {pkl_path} or {pkl_path}.gz')
                return None

            if pkl_path.endswith(".gz"):
                with gzip.open(pkl_path, 'rb') as f:  
                    raw_features = pickle.load(f)
            else:
                with open(pkl_path, 'rb') as f:
                    raw_features = pickle.load(f)

            assert raw_features['aatype'].shape[0] == len(seq), f"cached feat shape {raw_features['aatype'].shape} missmatch with seq {len(seq)}"

            logger.info(f'[MSA/Template] {desc}; cache matched in {pkl_path}')
            return chain_id, raw_features, desc, seq
        else:
            logger.info(f'[MSA/Template] {desc}; no exact sequence matched in {cache_dict_path}')
    except Exception as e:
        logger.error(f'[MSA/Template] {desc}; error when searching cache {cache_dict_path}: {str(e)}')
        return None
    return None
         

def update_dynamic_cache(cache_dict_path, seq, pkl_dir, worker_id="", max_retries=300, retry_delay=0.1):
    """
    使用文件锁机制安全更新缓存，避免多进程并发访问时的数据丢失
    支持重试机制，适合 slurm 抢占/重排环境
    """
    # 创建锁文件路径
    lock_file = f"{cache_dict_path}.lock"
    if not os.path.exists(lock_file):
        with open(lock_file, 'w') as f:
            pass
    os.chmod(lock_file, 0o777)

    logger.info(f"update_dynamic_cache updating {cache_dict_path} (worker: {worker_id})")
    
    max_num_cache = 200000
    temp_file = None
    
    def cleanup_temp_file():
        """清理临时文件"""
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)
            logger.debug(f"Cleaned up temp file: {temp_file}")
    
    with open(lock_file, 'a') as f:
        def signal_handler(signum, frame):
            """信号处理器，确保清理临时文件和释放锁"""
            logger.info(f"update_dynamic_cache {worker_id} received signal {signum}, cleaning up...")
            cleanup_temp_file()
            fcntl.flock(f, fcntl.LOCK_UN)
            # 重新抛出信号，让程序正常退出
            if signum == signal.SIGTERM:
                raise KeyboardInterrupt("Received SIGTERM")
        
        original_sigterm = signal.signal(signal.SIGTERM, signal_handler)
        
        for attempt in range(max_retries):
            # Try to acquire non-blocking lock
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                logger.info(f"update_dynamic_cache {worker_id} lock acquired, attempt {attempt}")
                break
            except (IOError, OSError):
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    logger.error(f"Failed to acquire lock after {max_retries} attempts")
                    return      # 如果获取锁失败，放弃更新缓存，直接返回
        
        print(f"update_dynamic_cache {worker_id} lock acquired, {seq}, {pkl_dir}", file=f)
        
        # 打印环境变量信息，for debug
        # project_root = os.environ.get('PROJECT_ROOT', 'NOT_SET')
        # slurm_job_id = os.environ.get('SLURM_JOB_ID', 'NOT_SET')
        # slurm_submit_dir = os.environ.get('SLURM_SUBMIT_DIR', 'NOT_SET')
        # print(f"Environment info - PROJECT_ROOT: {project_root}, SLURM_JOB_ID: {slurm_job_id}, SLURM_SUBMIT_DIR: {slurm_submit_dir}, PID: {os.getpid()}", file=f)
        
        try:
            # 步骤2: 读取现有缓存数据，注意 CFS 上的文件一致性问题
            seq_to_pkl = OrderedDict()
            last_size = -100
            for attempt in range(max_retries):
                os.system(f"ls -l {os.path.dirname(cache_dict_path)} > /dev/null 2>&1")
                if not os.path.exists(cache_dict_path):
                    time.sleep(retry_delay)
                    continue

                current_size = os.path.getsize(cache_dict_path)
                if current_size != last_size:
                    last_size = current_size
                    time.sleep(retry_delay)
                    continue
                else:
                    # 到这里，文件大小稳定，可以读取
                    with open(cache_dict_path, 'r') as f_cache:
                        seq_to_pkl = json.load(f_cache, object_pairs_hook=OrderedDict)
                    logger.info(f"update_dynamic_cache {len(seq_to_pkl)} keys loaded from {cache_dict_path}")
                    break
        
            # 步骤3: 更新数据
            clean_seq = seq.replace("\n", "")
            clean_pkl_dir = pkl_dir.replace("\n", "")
            seq_to_pkl[clean_seq] = clean_pkl_dir
            while len(seq_to_pkl) > max_num_cache:
                seq_to_pkl.popitem(last=False)
            logger.info(f"update_dynamic_cache {len(seq_to_pkl)} keys after update")
            
            # 步骤4: 创建临时文件
            cache_dir = os.path.dirname(cache_dict_path)
            if not cache_dir:
                cache_dir = "."
            temp_fd, temp_file = tempfile.mkstemp(
                dir=cache_dir,
                prefix=f".cache_update_{worker_id}_{os.getpid()}_",
                suffix='.tmp'
            )
            logger.debug(f"Created temp file: {temp_file}")
            
            # 步骤5: 写入临时文件
            with os.fdopen(temp_fd, 'w') as f_t:
                json.dump(seq_to_pkl, f_t, indent=2)
                f_t.flush()
                os.fsync(f_t.fileno())  # 确保数据写入磁盘
            os.chmod(temp_file, 0o777)
            
            # 步骤6: 原子替换
            os.replace(temp_file, cache_dict_path)
            fcntl.flock(f, fcntl.LOCK_UN)
            logger.info(f"update_dynamic_cache {worker_id} successfully updated cache")
        finally:
            signal.signal(signal.SIGTERM, original_sigterm)
            cleanup_temp_file()


def search_cache(msa_index_fasta, binary_path, desc, seq, chain_id, feat_dir):
   """ 
   search seq from given protein_msa_index_fasta and feat_dir
   """
   t0 = time.time()
   cache_hit = None
   logger.info(f'[MSA/Template] {desc}; fuzzily searching cache from {msa_index_fasta}')
   if not os.path.exists(feat_dir):
      logger.info(f'[MSA/Template] {desc}; feat_dir {feat_dir} not exists')
      return cache_hit
   if not os.path.exists(msa_index_fasta):
      logger.info(f'[MSA/Template] {desc}; msa_index_fasta {msa_index_fasta} not exists')
      return cache_hit
   
   pdb_chainid_2_seq = {}
   for line in open(msa_index_fasta):
      if line.startswith(">"): pdb_chainid = re.findall(r'[\w\d]{4,}_[\w\d]+', line)[0]
      else: pdb_chainid_2_seq[pdb_chainid] = line.replace("\n", "")

   try:
      cache_jackhmmer = jackhmmer.Jackhmmer(binary_path=binary_path,
                                  database_path=msa_index_fasta, e_value=1e-10)
      
      with tempfile.NamedTemporaryFile(delete=True, mode='w', suffix='.fasta') as temp_file:
          temp_file.write(f">{desc}\n")
          temp_file.write(seq)
          temp_file.flush()
          res = cache_jackhmmer.query(temp_file.name, max_sequences=100)
      
      a3m_str = parsers.convert_stockholm_to_a3m(res[0]['sto'])

      if len(a3m_str.split('\n')) > 2:
          # 从 MSA 中找到一个符合缓存标准的序列，如果有，则直接使用缓存的特征文件
          query = a3m_str.split('\n')[1]
          seq_meta = a3m_str.split('\n')[2::2]
          seq_msa = a3m_str.split('\n')[3::2]
          for s_m, s in zip(seq_meta, seq_msa):
              cache_hit = None
              long_num, short_num, mismatch, long_pos, short_pos = count_msa_lsm(query, s)

              if len(query) <= 16 and query == s:    # peptide, precise match
                  cache_hit = s_m
                  break
              elif len(query) <= 64 and long_num + short_num <= 2 and mismatch <= 1:
                  # short chain, allow 1 mismatch and 2 long/short
                  cache_hit = s_m
              elif long_num + short_num <= max(math.ceil(0.003 * len(query)), 2) and \
                  mismatch <= max(math.ceil(0.0015 * len(query)), 2):
                  cache_hit = s_m
              
              # cache_hit 是命中序列的 meta, s是命中的序列。
              if cache_hit is not None:
                  logger.info(f'[MSA/Template] {desc}; cache hit! sequence meta: {cache_hit}')
                  try:
                      # read meta_data line
                      meta_list = re.split(r'\s+', s_m)
                      match_pdb_chainid, match_range = meta_list[0][1:].split('/')

                      # extract pdb_id(with 4 or more chr or digits) and chain_id(with 1 or more chr or digits)
                      match_pdb_chainid = re.findall(r'[\w\d]{4,}_[\w\d]+', match_pdb_chainid)

                      if not len(match_pdb_chainid) > 0:
                          raise FileNotFoundError(f'[MSA/Template] {desc}; cannot find pdb_chainid pattern in {match_pdb_chainid}')

                      match_pdb_chainid = match_pdb_chainid[0]
                      match_cache_seq = pdb_chainid_2_seq[match_pdb_chainid]
                      assert len(match_cache_seq) - len(seq) < 30, f"[MSA/Template] {desc}; cached seq {len(match_cache_seq)} too long for target seq {len(seq)}"

                      # read cached feature pkl
                      pkl_path = os.path.join(feat_dir, match_pdb_chainid, 'features.pkl.gz')
                      if not os.path.exists(pkl_path):
                          pkl_path = os.path.join(feat_dir, match_pdb_chainid, 'features.pkl')
                      if not os.path.exists(pkl_path):
                          raise FileNotFoundError(f'[MSA/Template] {desc}; cannot find cached feature pkl {pkl_path} or {pkl_path}.gz')
                     
                      if pkl_path.endswith(".gz"):
                         with gzip.open(pkl_path, 'rb') as f:
                          raw_features = pickle.load(f)
                      else:
                         with open(pkl_path, 'rb') as f:
                          raw_features = pickle.load(f)

                      # modify raw_features according to match_range
                      m_start, m_end = [int(i) for i in match_range.split('-')]
                      raw_features = cache_feat_mod(raw_features, m_start, m_end, long_pos, short_pos)
                      assert raw_features['aatype'].shape[0] == len(seq), f"cached feat shape {raw_features['aatype'].shape} missmatch with seq {len(seq)}"
                      logger.info(f'[MSA/Template] {desc}; cache msa processed successfully! {match_pdb_chainid}/{match_range}; use: {time.time() - t0}')
                      logger.info(f'[MSA/Template] {desc}; cache msa depth {raw_features["num_alignments"].max()}')

                      if 'template_all_atom_mask' in raw_features:                                                                                       
                          raw_features['template_all_atom_masks'] = raw_features.pop('template_all_atom_mask')
                      
                      return chain_id, raw_features, desc, seq
                  except Exception as e:
                      logger.warning(e.__class__.__name__)
                      logger.warning(e)
                      logger.warning(f'[MSA/Template] {desc}; try next one!')
                      cache_hit = None
   except Exception as e:
        logger.warning(e.__class__.__name__)
        logger.warning(e)
        logger.warning(f'[MSA/Template] {desc}; try next one!')

   return cache_hit


def count_msa_lsm(query, msa_seq):
    """
    count query's long/short/mismatch number compared with MSA seq
    """
    long_num = 0
    short_num = 0
    mismatch = 0
    long_pos = []
    short_pos = []

    # count query's long/short/mismatch number compared with MSA seq
    for i in range(len(query)):
        while msa_seq[i].islower():
            short_pos.append(i + short_num)
            short_num += 1
            msa_seq = msa_seq[:i] + msa_seq[i + 1:]

        if query[i] != '-' and msa_seq[i] != '-':
            if query[i] != msa_seq[i]:
                mismatch += 1
        elif query[i] != '-' and msa_seq[i] == '-':
            long_pos.append(i + short_num)
            long_num += 1

    return long_num, short_num, mismatch, long_pos, short_pos


def feat_mod(raw_features, insert_or_del, index):
    if index < 0: index = 0     # 需要操作序列头，所以统一改为0

    if insert_or_del == 'insert':
        raw_features['aatype'] = np.insert(raw_features['aatype'], index, values=0, axis=0)
        raw_features['between_segment_residues'] = np.insert(raw_features['between_segment_residues'], index, values=0, axis=0)
        raw_features['sequence'][0] = b'-' + raw_features['sequence'][0]
        raw_features['deletion_matrix_int'] = np.insert(raw_features['deletion_matrix_int'], index, values=0, axis=-1)
        raw_features['msa'] = np.insert(raw_features['msa'], index, values=residue_constants.restypes_with_x_and_gap.index('-'), axis=-1)
        raw_features['deletion_matrix_int_all_seq'] = np.insert(raw_features['deletion_matrix_int_all_seq'], index, values=0, axis=-1)
        raw_features['msa_all_seq'] = np.insert(raw_features['msa_all_seq'], index, values=residue_constants.restypes_with_x_and_gap.index('-'), axis=-1)
        raw_features['template_aatype'] = np.insert(raw_features['template_aatype'], index, values=0, axis=1)
        raw_features['template_all_atom_masks'] = np.insert(raw_features['template_all_atom_masks'], index, values=0, axis=1)
        raw_features['template_all_atom_positions'] = np.insert(raw_features['template_all_atom_positions'], index, values=0, axis=1)
        for i, seq in enumerate(raw_features['template_sequence']):
            raw_features['template_sequence'][i] = b'-' + seq
    else:
        raw_features['aatype'] = np.delete(raw_features['aatype'], index, axis=0)
        raw_features['between_segment_residues'] = np.delete(raw_features['between_segment_residues'], index, axis=0)
        raw_features['sequence'][0] = raw_features['sequence'][0][:index] + raw_features['sequence'][0][index+1:]
        raw_features['deletion_matrix_int'][:, index+1] = raw_features['deletion_matrix_int'][:, index+1] + raw_features['deletion_matrix_int'][:, index]
        raw_features['deletion_matrix_int'] = np.delete(raw_features['deletion_matrix_int'], index, axis=-1)
        raw_features['msa'] = np.delete(raw_features['msa'], index, axis=-1)
        raw_features['deletion_matrix_int_all_seq'][:, index+1] = raw_features['deletion_matrix_int_all_seq'][:, index+1] + raw_features['deletion_matrix_int_all_seq'][:, index]
        raw_features['deletion_matrix_int_all_seq'] = np.delete(raw_features['deletion_matrix_int_all_seq'], index, axis=-1)
        raw_features['msa_all_seq'] = np.delete(raw_features['msa_all_seq'], index, axis=-1)
        raw_features['template_aatype'] = np.delete(raw_features['template_aatype'], index, axis=1)
        raw_features['template_all_atom_masks'] = np.delete(raw_features['template_all_atom_masks'], index, axis=1)
        raw_features['template_all_atom_positions'] = np.delete(raw_features['template_all_atom_positions'], index, axis=1)
        for i, seq in enumerate(raw_features['template_sequence']):
            raw_features['template_sequence'][i] = seq[:index] + seq[index+1:]

    return raw_features


def cache_feat_mod(raw_features, m_start, m_end, long_pos, short_pos):
    m_start -= 1        # a3m index starts from 1
    temp_new_features = raw_features.copy()

    # cut feature according to current matched range of cached sequence
    temp_new_features['aatype'] = raw_features['aatype'][m_start:m_end]
    temp_new_features['between_segment_residues'] = raw_features['between_segment_residues'][m_start:m_end]
    temp_new_features['sequence'][0] = raw_features['sequence'][0][m_start:m_end]
    temp_new_features['deletion_matrix_int'] = raw_features['deletion_matrix_int'][..., m_start:m_end]
    temp_new_features['deletion_matrix_int_all_seq'] = raw_features['deletion_matrix_int_all_seq'][..., m_start:m_end]
    temp_new_features['msa'] = raw_features['msa'][..., m_start:m_end]
    temp_new_features['msa_all_seq'] = raw_features['msa_all_seq'][..., m_start:m_end]
    temp_new_features['template_sequence'][0] = raw_features['template_sequence'][0][m_start:m_end]
    temp_new_features['template_aatype'] = raw_features['template_aatype'][:, m_start:m_end, ...]
    temp_new_features['template_all_atom_masks'] = raw_features['template_all_atom_masks'][:, m_start:m_end, ...]
    temp_new_features['template_all_atom_positions'] = raw_features['template_all_atom_positions'][:, m_start:m_end, ...]
    
    long_pos = np.array([i - m_start for i in long_pos if i < m_end])
    short_pos = np.array([i - m_start for i in short_pos if i < m_end])

    i = len(long_pos) - 1
    j = len(short_pos) - 1
    while i >= 0 and j >= 0:
        if long_pos[i] > short_pos[j]:
            temp_new_features = feat_mod(temp_new_features, 'insert', long_pos[i])
            i -= 1
        else:
            temp_new_features = feat_mod(temp_new_features, 'delete', short_pos[j])
            j -= 1
    while i >= 0:
        temp_new_features = feat_mod(temp_new_features, 'insert', long_pos[i])
        i -= 1
    while j >= 0:
        temp_new_features = feat_mod(temp_new_features, 'delete', short_pos[j])
        j -= 1

    temp_new_features['residue_index'] = np.arange(len(temp_new_features['aatype']))
    temp_new_features['seq_length'] = np.full(len(temp_new_features['aatype']), len(temp_new_features['aatype']), dtype='int32')
    temp_new_features['num_alignments'] = np.full(len(temp_new_features['aatype']), len(temp_new_features['msa']), dtype='int32')
    return temp_new_features
