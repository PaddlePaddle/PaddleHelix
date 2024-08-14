from concurrent import futures
import glob
import os
import subprocess
from typing import Any, Callable, Mapping, Optional, Sequence
from urllib import request
from absl import logging
from Bio import SeqIO

from helixfold.data import parsers
from helixfold.data.tools import utils


class Nhmmer:
    """Python wrapper of the Hmmer binary."""
    def __init__(self,
               *,
               binary_path: str,
               database_path: str,
               n_cpu: int = 8,
               e_value: float = 0.001,
               get_tblout: bool = False,
               num_streamed_chunks: Optional[int] = None,
               streaming_callback: Optional[Callable[[int], None]] = None):

        self.binary_path = binary_path
        self.database_path = database_path
        self.num_streamed_chunks = num_streamed_chunks

        if not os.path.exists(self.database_path) and num_streamed_chunks is None:
            logging.error('Could not find Jackhmmer database %s', database_path)
            raise ValueError(f'Could not find Jackhmmer database {database_path}')

        self.n_cpu = n_cpu
        self.e_value = e_value
        self.get_tblout = get_tblout
        self.streaming_callback = streaming_callback


    def _query_chunk(self,
              input_fasta_path: str,
              database_path: str,
              max_sequences: Optional[int] = None) -> Sequence[Mapping[str, Any]]:

        """Queries the database chunk using Hmmer."""
        query_length = 0
        # Get the length of the query sequence
        for record in SeqIO.parse(input_fasta_path, "fasta"):
            sequence = record.seq
            query_length = len(sequence)

        with utils.tmpdir_manager() as query_tmp_dir:
            sto_path = os.path.join(query_tmp_dir, 'output.sto')

            cmd_flags = [
                # Don't pollute stdout with nhmmer output.
                '-o', '/dev/null',
                '-A', sto_path,
                '--noali',
                '--incE', str(self.e_value),
                '--F3', str(0.00005) if query_length >= 50 else str(0.02),
                # Report only sequences with E-values <= x in per-sequence output.
                '-E', str(self.e_value),
                '--cpu', str(self.n_cpu),
                '--rna',
                '--watson'
            ]
            if self.get_tblout:
                tblout_path = os.path.join(query_tmp_dir, 'tblout.txt')
                cmd_flags.extend(['--tblout', tblout_path])

            cmd = [self.binary_path] + cmd_flags + [input_fasta_path,
                                                    database_path]

            logging.info('Launching subprocess "%s"', ' '.join(cmd))

            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            with utils.timing(
                f'nhmmer ({os.path.basename(database_path)}) query'):
                _, stderr = process.communicate()
                retcode = process.wait()

            if retcode:
                raise RuntimeError(
                    'nhmmer failed\nstderr:\n%s\n' % stderr.decode('utf-8'))

            # Get e-values for each target name
            tbl = ''
            if self.get_tblout:
                with open(tblout_path) as f:
                    tbl = f.read()

            # add the query sequence to sto

            if max_sequences is None:
                with open(sto_path) as f:
                    sto = f.read()
            else:
                sto = parsers.truncate_stockholm_msa(sto_path, max_sequences)

        raw_output = dict(
            sto=sto,
            tbl=tbl,
            stderr=stderr,
            e_value=self.e_value)

        return raw_output

    def query(self,
            input_fasta_path: str,
            max_sequences: Optional[int] = None) -> Sequence[Mapping[str, Any]]:
        """Queries the database using Jackhmmer."""
        if self.num_streamed_chunks is None:
            single_chunk_result = self._query_chunk(
                input_fasta_path, self.database_path, max_sequences)
            return [single_chunk_result]

        db_basename = os.path.basename(self.database_path)
        db_remote_chunk = lambda db_idx: f'{self.database_path}.{db_idx}'
        db_local_chunk = lambda db_idx: f'/tmp/ramdisk/{db_basename}.{db_idx}'

        # Remove existing files to prevent OOM
        for f in glob.glob(db_local_chunk('[0-9]*')):
            try:
                os.remove(f)
            except OSError:
                print(f'OSError while deleting {f}')

        # Download the (i+1)-th chunk while Jackhmmer is running on the i-th chunk
        with futures.ThreadPoolExecutor(max_workers=2) as executor:
            chunked_output = []
            for i in range(1, self.num_streamed_chunks + 1):
                # Copy the chunk locally
                if i == 1:
                    future = executor.submit(
                        request.urlretrieve, db_remote_chunk(i), db_local_chunk(i))
                if i < self.num_streamed_chunks:
                    next_future = executor.submit(
                        request.urlretrieve, db_remote_chunk(i+1), db_local_chunk(i+1))

                # Run Jackhmmer with the chunk
                future.result()
                chunked_output.append(self._query_chunk(
                    input_fasta_path, db_local_chunk(i), max_sequences))

                # Remove the local copy of the chunk
                os.remove(db_local_chunk(i))
                # Do not set next_future for the last chunk so that this works even for
                # databases with only 1 chunk.
                if i < self.num_streamed_chunks:
                    future = next_future
                if self.streaming_callback:
                    self.streaming_callback(i)

        return chunked_output

        





