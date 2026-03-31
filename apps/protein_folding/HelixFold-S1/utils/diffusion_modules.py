"""modules for diffusion."""

import math
import paddle
import paddle.nn as nn
import numpy as np
from helixfold.model import geometry, all_atom, r3
from utils.igso3 import IGSO3
from scipy.spatial.transform import Rotation as scipy_R
import copy

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = v[t]
    return np.reshape(out, [t.shape[0]] + [1] * (len(x_shape) - 1))


class Swish(nn.Layer):
    """ Swish activation. """
    def forward(self, x):
        """tbd."""
        return x * nn.functional.sigmoid(x)


class TimeEmbedding(nn.Layer):
    """TimeEmbedding.
    """
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = np.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = np.exp(-emb)
        pos = np.arange(T).astype('float32')
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = np.stack([np.sin(emb), np.cos(emb)], axis=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = np.reshape(emb, (T, d_model))
        emb = paddle.to_tensor(emb.astype('float32'))

        init_w = nn.initializer.XavierUniform()
        init_b = nn.initializer.Constant(value=0.0)
        self.timembedding = nn.Sequential(
            nn.Embedding(num_embeddings=T, embedding_dim=d_model), 
            nn.Linear(d_model, dim, 
                weight_attr=paddle.ParamAttr(initializer=init_w),
                bias_attr=paddle.ParamAttr(initializer=init_b)
            ),
            Swish(),
            nn.Linear(d_model, dim, 
                weight_attr=paddle.ParamAttr(initializer=init_w),
                bias_attr=paddle.ParamAttr(initializer=init_b)
            ),
        )
        self.timembedding._sub_layers['0'].weight.set_value(emb)

    def forward(self, t):
        """tbd."""
        emb = self.timembedding(t)
        return emb


class Diffuser:
    """Diffuser.
    """
    def __init__(self, T, beta_1, beta_T):
        self.T = T

        self.betas = np.linspace(beta_1, beta_T, self.T)
            
        alphas = 1. - self.betas
        alphas_bar = np.cumprod(alphas, axis=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_bar = np.sqrt(alphas_bar)
        self.sqrt_one_minus_alphas_bar = np.sqrt(1. - alphas_bar)
        self.igso3 = IGSO3(
            T=T,
            min_sigma=0.02,
            max_sigma=1.5,
            schedule="linear",
            min_b=1.5,
            max_b=2.5,
            cache_dir="log/igso3",
            L=2000,
        )

    def last_timestep(self, batch_size):
        """ Return array of T with batch_size length """
        return np.array([self.T-1] * batch_size)

    def sample_timestep(self, batch_size):
        """ Sample t from T for batch_size times"""
        return np.random.randint(1, self.T, size=(batch_size, ))

    def diffuse_trans(self, x_0, t):
        """ Add t steps of noise to x_0 [batch, num_res, 3] """
        # Draw samples from a standard Normal distribution
        noise = np.random.standard_normal(size=x_0.shape) * 20
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        return x_t

    def diffuse_rots(self, x_0, t):
        """ Add t steps of noise to x_0 [batch, num_res, 3, 3] """
        return self.igso3.diffuse_rots(x_0, t)


class DDIMSampler:
    """DDIMSampler.
    """
    def __init__(self, T, beta_1, beta_T, 
                S, eta, # new params for DDIM
                igso3_diffuser,
                var_type='fixedlarge'):
        mean_type='xstart'
        assert mean_type in ['xprev', 'xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()
        # self.model = model
        self.T = T
        self.S = S
        # self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type
        full_time_steps = list(range(T))
        self.betas = np.linspace(beta_1, beta_T, T)

        sampled_time_steps = [x[0] for x in np.array_split(full_time_steps, S)]
        if not T - 1 in sampled_time_steps:
            sampled_time_steps.append(T - 1)
        self.sampled_time_steps = sampled_time_steps
        print('DDIM sampled_time_steps:', sampled_time_steps)

        ddpm_alphas = 1. - self.betas
        alphas = np.cumprod(ddpm_alphas, axis=0)
        alphas = alphas[sampled_time_steps]
        alphas_prev = np.pad(alphas, [1, 0], mode='constant', constant_values=1)[:-1]
        # posterior_var = self.betas * (1. - alphas_prev) / (1. - alphas)
        posterior_var = (1. - alphas_prev) / (1. - alphas) * (1 - alphas / alphas_prev)
        posterior_var *= eta ** 2

        self.alphas = alphas
        self.alphas_prev = alphas_prev
        self.posterior_var = posterior_var
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.posterior_log_var_clipped = np.log(
                np.concatenate([posterior_var[1:2], posterior_var[1:]]))
        self.sigmas = posterior_var
        self.so3_diffuser = igso3_diffuser
    
    def predict_xstart_from_eps(self, x_t, t, eps):
        """predict_xstart_from_eps."""
        def _get_coef_t(coef_tensor):
            return extract(coef_tensor, t, x_t.shape)

        assert eps.shape == x_t.shape
        return (x_t - np.sqrt(1 - _get_coef_t(self.alphas)) * eps) / \
                np.sqrt(_get_coef_t(self.alphas))

    def predict_eps_from_xstart(self, x_0, x_t, t):
        """predict_eps_from_xstart adapted from predict_xstart_from_eps."""
        def _get_coef_t(coef_tensor):
            return extract(coef_tensor, t, x_t.shape)

        assert x_0.shape == x_t.shape
        return (x_t - x_0 * np.sqrt(_get_coef_t(self.alphas))) / \
                np.sqrt(1 - _get_coef_t(self.alphas))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, eps)
        """
        def _get_coef_t(coef_tensor):
            return extract(coef_tensor, t, x_t.shape)

        assert x_0.shape == x_t.shape
        eps = self.predict_eps_from_xstart(x_0, x_t, t)

        posterior_mean = np.sqrt(_get_coef_t(self.alphas_prev)) * x_0 + \
                np.sqrt(1 - _get_coef_t(self.alphas_prev) - _get_coef_t(self.sigmas) ** 2) * eps

        posterior_log_var_clipped = _get_coef_t(self.posterior_log_var_clipped)
        return posterior_mean, posterior_log_var_clipped

    def p_mean_variance(self, x_t, t, x_0):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': np.log(np.concatenate([self.posterior_var[1:2],
                                                self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        return model_mean, model_log_var

    def get_next_trans(self, x_0_trans, x_t_trans, t, time_step):
        mean, log_var = self.p_mean_variance(x_t=x_t_trans, t=t, x_0=x_0_trans)
        # no noise when t == 0
        if time_step > 0:
            noise = paddle.to_tensor(np.random.standard_normal(size=x_t_trans.shape)) * 20
        else:
            noise = 0
        x_t_trans = mean + np.exp(0.5 * log_var) * noise  # re-parameterization version of torch.normal(mean, torch.sqrt(log_var * noise)) 
        return x_t_trans

    def get_next_rots(self, x_0_rots, x_t_rots, t, time_step, noise_scale=0.5):
        # ref: RFDiffusion rfdiffusion/inference/utils.py
        if time_step > 0:
            # TODO: verify if scipy_R convert is needed
            R_0 = scipy_R.from_matrix(x_0_rots.squeeze().numpy()).as_matrix()
            R_t = scipy_R.from_matrix(x_t_rots.squeeze().numpy()).as_matrix()

            L = R_t.shape[0]
            all_rot_transitions = np.broadcast_to(np.identity(3), (L, 3, 3)).copy()
            # Sample next frame for each residue
            if True: # so3_type == "igso3":
                # don't do calculations on masked positions since they end up as identity matrix
                all_rot_transitions = self.so3_diffuser.reverse_sample_vectorized(
                    R_t,
                    R_0,
                    t,
                    noise_level=noise_scale,
                    mask=None,
                    return_perturb=True,
                ) 
            all_rot_transitions = all_rot_transitions[None, :, :, :] # [1, num_res, 3, 3]


            # Apply the interpolated rotation matrices to the coordinates
            x_t_rots = (
                np.einsum(
                    "lrij,lraj->lrai",
                    all_rot_transitions,
                    x_t_rots.numpy(),
                    )
                ).astype('float32')
            return paddle.to_tensor(x_t_rots)
        else:
            # return x_t_rots
            return x_0_rots # TODO: check rot denoise eq

    def inference(self, x_T_trans, x_T_rots,
        batch, helixfold, repeated_size, compute_loss, position_scale):
        x_t_trans, x_t_rots = x_T_trans, x_T_rots
        print(f"init x_t_tarns std: {float(x_t_trans.std())}  mean: {float(x_t_trans.mean())} shape: {x_t_trans.shape} tensor:{x_t_trans[:,:5,:]} numpy:{x_t_trans[:,:5,:].numpy()}")
        print(f"init x_t_rots std: {float(x_t_rots.std())}  mean: {float(x_t_rots.mean())}  shape: {x_t_rots.shape} tensor:{x_t_rots[:,:2,:]} numpy:{x_t_rots[:,:2,:].numpy()} ")
        for time_step in reversed(range(self.S)):
            real_t = self.sampled_time_steps[time_step]
            t = paddle.to_tensor(np.array([1] * x_t_trans.shape[0], dtype='int32') * time_step)
            real_t = paddle.to_tensor(np.array([1] * x_t_trans.shape[0], dtype='int32') * real_t)
            print(f"Sample step {time_step}[{real_t}] of {self.S}[{self.T}]")

            """
            Algorithm 2.
            """
            batch_updated = copy.deepcopy(batch)
            batch_updated["feat"].update({"trans_diffused": 
                    paddle.repeat_interleave(x_t_trans.unsqueeze(1), repeated_size, axis=1)
                    })
            batch_updated["feat"].update({"rots_diffused": 
                paddle.repeat_interleave(x_t_rots.unsqueeze(1), repeated_size, axis=1)
                    })
            batch_updated["feat"].update({"diffuse_timestep": 
                    paddle.repeat_interleave(real_t.unsqueeze(1), repeated_size, axis=1)  # repeat to avoid batch ensemble error
                    })
            res = helixfold(             
                batch_updated['feat'],
                batch_updated['label_cropped'],
                ensemble_representations=True,
                return_representations=True,
                compute_loss=compute_loss)
            value = res['structure_module']
            x_0_trans = geometry.Rigid3Array.from_array(value['final_affines']).translation.to_array()
            x_0_trans /= position_scale # run sample with unscaled affine
            x_0_rots = geometry.Rigid3Array.from_array(value['final_affines']).rotation.to_array()

            x_t_trans = self.get_next_trans(x_0_trans, x_t_trans, t, time_step)
            x_t_rots = self.get_next_rots(x_0_rots, x_t_rots, 1, time_step)
            print(f"step {time_step} x_t_tarns std:{float(x_t_trans.std())} mean:{float(x_t_trans.mean())} shape:{x_t_trans.shape} tensor:{x_t_trans[:,:5,:]} numpy:{x_t_trans[:,:5,:]}")
            print(f"step {time_step} x_t_rots std:{float(x_t_rots.std())} mean:{float(x_t_rots.mean())} shape:{x_t_rots.shape} tensor:{x_t_rots[:,:2,:]} numpy:{x_t_rots[:,:2,:]}")

        x_0_trans = x_t_trans
        x_0_rots = x_t_rots
        # torch.clip(x_0, -1, 1)
        batch0 = {k: v[:, 0] for k, v in batch['feat'].items()}
        return update_res(res, batch0, position_scale, new_trans=x_0_trans, new_rot=x_0_rots, new_angles=None)



def update_res(res, batch0, position_scale, new_trans=None, new_rot=None, new_angles=None):
    """Update res['structure_module']['final_atom_positions'] with new new_trans/rot/angles.
    """
    value = res["structure_module"]
    aatype = batch0["aatype"]  # (B, N)
    final_affines = geometry.Rigid3Array.from_array(value['final_affines'])
    pd_trans, pd_rot = final_affines.translation, final_affines.rotation
    angles = value['sidechains']['angles_sin_cos'][-1]  # last layer angles

    if new_rot is not None:
        pd_rot = geometry.Rot3Array.from_array(new_rot)
    if new_trans is not None:
        pd_trans = geometry.Vec3Array.from_array(new_trans)
    if new_angles is not None:
        angles = new_angles

    # Adapted from StructureModule.forward
    affine = geometry.Rigid3Array(pd_rot, pd_trans) # update trans/rot if provided
    affine = affine.scale_translation(position_scale) # already done before final_affines
    affine_translation = affine.translation.to_array()
    print(f"final affine_translation: std:{float(affine_translation.std())}  mean:{float(affine_translation.mean())} shape:{affine_translation.shape} tensor:{affine_translation[:,:5,:]} numpy:{affine_translation[:,:5,:].numpy()}")

    # Adapted from MultiRigidSidechain.forward
    # use_new_affine = self.global_config.get('multimer_mode', False)  # ref to _generate_affines
    rot = r3.Rots(affine.rotation.to_array())
    vec = r3.Vecs(affine.translation.to_array())
    backbone_to_global = r3.Rigids(rot, vec)

    all_frames_to_global = all_atom.torsion_angles_to_frames(
                aatype, backbone_to_global, angles)
    pred_positions = all_atom.frames_and_literature_positions_to_atom14_pos(
            aatype, all_frames_to_global)

    # Adapted from StructureModule.forward
    # (B, N, 14, 3)
    atom14_pred_positions = pred_positions.translation # output['sc']['atom_pos'][-1]
    res["structure_module"]['final_atom14_positions'] = atom14_pred_positions
    # (B, N, 37, 3)
    atom37_pred_positions = all_atom.atom14_to_atom37(
        atom14_pred_positions, batch0)
    atom37_pred_positions *= paddle.unsqueeze(
        batch0['atom37_atom_exists'], axis=-1)
    res["structure_module"]['final_atom_positions'] = atom37_pred_positions
    return res