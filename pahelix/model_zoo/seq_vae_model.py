#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This is an implementation of sequence VAE from:
https://github.com/ molecularsets/moses
"""

import paddle
from paddle.io import Dataset 
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.dygraph as dg
import paddle.nn.functional as F
import paddle.nn as nn
from paddle.optimizer.lr import LRScheduler
import pdb


class VAE(nn.Layer):
    """The sequence VAE model

    Args:
        vocab: the vocab object.
        model_config: the json files of model parameters.
    """
    def __init__(self, vocab, model_config):
        super(VAE, self).__init__()

        self.config = model_config
            
        self.vocabulary = vocab
        # Special symbols
        for ss in ('bos', 'eos', 'unk', 'pad'):
            setattr(self, ss, getattr(vocab, ss))
        
        # Word embeddings layer
        n_vocab, d_emb = len(vocab), vocab.vectors.shape[1]
        self.x_emb = nn.Embedding(n_vocab, d_emb, self.pad)
        self.x_emb.weight.set_value(paddle.to_tensor(vocab.vectors))
        if self.config['freeze_embeddings']:        
            self.x_emb.weight.stop_gradient=True
                       
        # encoder
        self.encoder_rnn = nn.GRU(
                d_emb,
                self.config['q_d_h'],
                num_layers=self.config['q_n_layers'],                
                dropout=self.config['q_dropout'] if self.config['q_n_layers'] > 1 else 0,
                direction= 'bidirectional'  if self.config['q_bidir'] else 'forward'
            )
        
        q_d_last = self.config['q_d_h'] * (2 if self.config['q_bidir'] else 1)
        self.q_mu = nn.Linear(q_d_last, self.config['d_z'])
        self.q_logvar = nn.Linear(q_d_last, self.config['d_z'])
               
        # decoder        
        self.decoder_rnn = nn.GRU(
                d_emb + self.config['d_z'],
                self.config['d_d_h'],
                num_layers=self.config['d_n_layers'],
                dropout=self.config['d_dropout'] if self.config['d_n_layers'] > 1 else 0
            )
        
        self.decoder_lat = nn.Linear(self.config['d_z'], self.config['d_d_h'])
        self.decoder_fc = nn.Linear(self.config['d_d_h'], n_vocab)
        
    def forward(self, x):
        """
        Model forward
        """

        # Encoder: x -> z, kl_loss
        z, kl_loss, _, = self.forward_encoder(x)
        
        # Decoder: x, z -> recon_loss
        recon_loss = self.forward_decoder(x, z)

        return kl_loss, recon_loss 
        
    def forward_encoder(self, x):
        """
        encoder
        """
        
        data = x[0]
        data_length = x[1]
        
    
        ######### embedding
        embedding_data = self.x_emb(data)
                                   
        ######### GRU encoder
        _, h = self.encoder_rnn(embedding_data, sequence_length=data_length)
        
        h = h[-(1 + int(self.config['q_bidir'])):]       
        h = paddle.concat(h.split(1 + int(self.config['q_bidir'])), axis=-1).squeeze(0)
                
        mu, logvar = self.q_mu(h), self.q_logvar(h)           
        eps = paddle.randn(shape=mu.shape) 
        z = mu + (logvar / 2).exp() * eps                
                
        kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()

        return (z, kl_loss, mu)
    
    def forward_decoder(self, x, z):
        """
        decoder
        """
        data = x[0]
        data_length = x[1]
        
        
        embedding_data = self.x_emb(data)
                
        z_0 = paddle.expand(z.unsqueeze(1), shape=[z.unsqueeze(1).shape[0], \
                embedding_data.shape[1], z.unsqueeze(1).shape[2]])
        
        x_input = paddle.concat([embedding_data, z_0], axis=-1)
        
        
        h_0 = self.decoder_lat(z)                
        h_0 = paddle.expand(h_0.unsqueeze(0), \
            shape=[self.decoder_rnn.num_layers, h_0.unsqueeze(0).shape[1], h_0.unsqueeze(0).shape[2]])
        
        ####
        output, _ = self.decoder_rnn(x_input, h_0, sequence_length=data_length)
        y = self.decoder_fc(output)
        
        recon_loss = F.cross_entropy(paddle.reshape(y[:, :-1], shape=[-1, y.shape[-1]]), \
            paddle.reshape(data[:, 1:], shape=[-1]), \
            ignore_index=self.pad
        )
        
        return recon_loss

    def sample_z_prior(self, n_batch):
        """Sampling z ~ p(z) = N(0, I)

        Args:
            n_batch: number of batches

        Returns: 
        (n_batch, d_z) of floats, sample of latent z
        """
        return paddle.randn([n_batch, self.q_mu.weight.shape[1]])
    
    def tensor2string(self, tensor):
        """
        convert tensor values to sequence string
        """
        ids = tensor.numpy().tolist()
        string = self.vocabulary.ids2string(ids, rem_bos=True, rem_eos=True)

        return string
            
    def sample(self, n_batch, max_len=100, z=None, temp=1.0):
        """Generating n_batch samples in eval mode (`z` could be
            not on same device)

        Args:
            n_batch: number of sentences to generate
            max_len: max len of samples
            z: (n_batch, d_z) of floats, latent vector z or None
            temp: temperature of softmax

        Returns: 
            list of tensors of strings, samples sequence x
        """
        
        if z is None:
            z = self.sample_z_prior(n_batch)
        z_0 = z.unsqueeze(1)
        
        # Initial values
        h = self.decoder_lat(z)
        h = paddle.expand(h.unsqueeze(0), \
            shape=[self.decoder_rnn.num_layers, h.unsqueeze(0).shape[1], h.unsqueeze(0).shape[2]])
        
        w = paddle.expand(paddle.to_tensor(self.bos), shape=[n_batch])
        x = paddle.expand(paddle.to_tensor([self.pad]), shape=[n_batch, max_len])
        
        x[:, 0] = self.bos
        
        end_pads = paddle.expand(paddle.to_tensor([max_len]), shape=[n_batch])
        eos_mask = paddle.zeros([n_batch], dtype='bool')
        
        # Generating cycle
        for i in range(1, max_len):
            x_emb = self.x_emb(w).unsqueeze(1)            
            x_input = paddle.concat([x_emb, z_0], axis=-1)
            
            o, h = self.decoder_rnn(x_input, h)
            y = self.decoder_fc(o.squeeze(1))
            y = F.softmax(y / temp, axis=-1)
            
            w = paddle.multinomial(y, 1)[:, 0]
            #w = paddle.argmax(y, 1)
           
            # convert to numpy in order to slice the mask
            x = x.numpy()
            eos_mask = eos_mask.numpy()
            w = w.numpy()
            end_pads = end_pads.numpy()
                       
            x[~eos_mask, i] = w[~eos_mask]
            i_eos_mask = ~ eos_mask & (w == self.eos)
            end_pads[i_eos_mask] = i + 1
            eos_mask = eos_mask | i_eos_mask
            
            # convert back to tensor
            x = paddle.to_tensor(x)
            w = paddle.to_tensor(w)
            eos_mask = paddle.to_tensor(eos_mask)
            end_pads = paddle.to_tensor(end_pads)
                        
        # Converting `x` to list of tensors
        new_x = []
        for i in range(x.shape[0]):
            new_x.append(x[i, :int(end_pads[i])])
            
        return [self.tensor2string(i_x) for i_x in new_x]
