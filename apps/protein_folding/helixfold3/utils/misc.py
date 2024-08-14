# copyright (c) 2024 PaddleHelix Authors. All Rights Reserve.
#
# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0
# International License (the "License");  you may not use this file  except
# in compliance with the License. You may obtain a copy of the License at
#
#    http://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
misc utils
"""

import logging

def set_logging_level(level):
    level_dict = {
        "NOTSET": logging.NOTSET,
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s',
        level=level_dict[level],
        datefmt='%Y-%m-%d %H:%M:%S')
    for h in logging.root.handlers: 
        h.setFormatter(
            logging.Formatter(
                '%(asctime)s %(levelname)s %(message)s', 
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
    logging.root.setLevel(level_dict[level])
