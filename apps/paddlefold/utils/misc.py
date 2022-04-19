# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
misc utils
"""

from collections import OrderedDict

__all__ = ['AverageMeter']


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Code was based on https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self, name='', fmt='f', postfix="", need_avg=True):
        self.name = name
        self.fmt = fmt
        self.postfix = postfix
        self.need_avg = need_avg
        self.reset()

    def reset(self):
        """ reset """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ update """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def total(self):
        """ total """
        return '{self.name}_sum: {self.sum:{self.fmt}}{self.postfix}'.format(
            self=self)

    @property
    def total_minute(self):
        """ total minute"""
        return '{self.name} {s:{self.fmt}}{self.postfix} min'.format(
            s=self.sum / 60, self=self)

    @property
    def mean(self):
        """ mean """
        return '{self.name}: {self.avg:{self.fmt}}{self.postfix}'.format(
            self=self) if self.need_avg else ''

    @property
    def value(self):
        """ value """
        return '{self.name}: {self.val:{self.fmt}}{self.postfix}'.format(
            self=self)


class TrainLogger(object):
    """ A warpper of training logger"""
    def __init__(self):
        self.info = OrderedDict()
        self.info['loss'] = AverageMeter("loss", ".5f", postfix=", ")
        self.info['reader_cost'] = AverageMeter("reader_cost", ".5f", postfix="s, ")
        self.info['forward_cost'] = AverageMeter("forward_cost", ".5f", postfix="s, ")
        self.info['backward_cost'] = AverageMeter("backward_cost", ".5f", postfix="s, ")
        self.info['gradsync_cost'] = AverageMeter("gradsync_cost", ".5f", postfix="s, ")
        self.info['update_cost'] = AverageMeter("update_cost", ".5f", postfix="s, ")
        self.info['batch_cost'] = AverageMeter("batch_cost", ".5f", postfix="s, ")
        self.info['avg_loss'] = AverageMeter("avg_loss", ".5f", postfix=", ")
        self.info['protein'] = AverageMeter("protein", "d", postfix=", ")

    def update(self, key, value, n=1):
        """ update value by key """
        self.info[key].update(value, n=n)

    def reset(self, key=None):
        """ reset all the item if key==None, otherwise reset the item by key"""
        if key is None:
            for k in self.info:
                self.info[k].reset()
        else:
            self.info[key].reset()


    def mean(self, key):
        """ get mean value by key """
        return self.info[key].avg

    def sum(self, key):
        """ get sum value by key """
        return self.info[key].sum

    def state_dict(self):
        """ get state dict """
        state = {}
        for key in self.info:
            if 'protein' == key:
                state[key] = self.info[key].sum
            else:
                state[key] = self.info[key].avg
        state['ips'] = self.info["protein"].sum / self.info["batch_cost"].sum
        return state

    def msg(self):
        """ return string """
        log_msg = ''
        for key in self.info:
            if 'protein' == key:
                log_msg += self.info[key].total
            else:
                log_msg += self.info[key].mean
        
        log_msg += f"ips: {self.info['protein'].sum / self.info['batch_cost'].sum:.5f} protein/s"
        return log_msg
