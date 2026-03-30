"""SO(3) diffusion methods."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation as scipy_R
import logging
import os
from scipy.spatial.transform import Rotation
import pickle
import scipy.linalg


### First define geometric operations on the SO3 manifold

# hat map from vector space R^3 to Lie algebra so(3)
def hat(v):
    hat_v = torch.zeros([v.shape[0], 3, 3])
    hat_v[:, 0, 1], hat_v[:, 0, 2], hat_v[:, 1, 2] = -v[:, 2], v[:, 1], -v[:, 0]
    return hat_v + -hat_v.transpose(2, 1)

# Logarithmic map from SO(3) to R^3 (i.e. rotation vector)
def Log(R): return torch.tensor(Rotation.from_matrix(R.numpy()).as_rotvec())
    
# logarithmic map from SO(3) to so(3), this is the matrix logarithm
def log(R): return hat(Log(R))

# Exponential map from vector space of so(3) to SO(3), this is the matrix
# exponential combined with the "hat" map
def Exp(A): return torch.tensor(Rotation.from_rotvec(A.numpy()).as_matrix())

# Angle of rotation SO(3) to R^+
def Omega(R): return np.linalg.norm(log(R), axis=[-2, -1])/np.sqrt(2.)

def f_igso3(omega, t, L):
    """Truncated sum of IGSO(3) distribution.

    This function approximates the power series in equation 5 of
    "DENOISING DIFFUSION PROBABILISTIC MODELS ON SO(3) FOR ROTATIONAL
    ALIGNMENT"
    Leach et al. 2022

    This expression diverges from the expression in Leach in that here, sigma =
    sqrt(2) * eps, if eps_leach were the scale parameter of the IGSO(3).

    With this reparameterization, IGSO(3) agrees with the Brownian motion on
    SO(3) with t=sigma^2 when defined for the canonical inner product on SO3,
    <u, v>_SO3 = Trace(u v^T)/2

    Args:
        omega: i.e. the angle of rotation associated with rotation matrix
        t: variance parameter of IGSO(3), maps onto time in Brownian motion
        L: Truncation level
    """
    ls = torch.arange(L)[None]  # of shape [1, L]
    return ((2*ls + 1) * torch.exp(-ls*(ls+1)*t/2) *
             torch.sin(omega[:, None]*(ls+1/2)) / torch.sin(omega[:, None]/2)).sum(dim=-1)

def d_logf_d_omega(omega, t, L):
    omega = torch.tensor(omega, requires_grad=True)
    log_f = torch.log(f_igso3(omega, t, L))
    return torch.autograd.grad(log_f.sum(), omega)[0].numpy()

# IGSO3 density with respect to the volume form on SO(3)
def igso3_density(Rt, t, L):
    return f_igso3(torch.tensor(Omega(Rt)), t, L).numpy()

def igso3_density_angle(omega, t, L): 
    return f_igso3(torch.tensor(omega), t, L).numpy()*(1-np.cos(omega))/np.pi

# grad_R log IGSO3(R; I_3, t)
def igso3_score(R, t, L):
    omega = Omega(R)
    unit_vector = np.einsum('Nij,Njk->Nik', R, log(R))/omega[:, None, None]
    return unit_vector * d_logf_d_omega(omega, t, L)[:, None, None]

def calculate_igso3(*, num_sigma, num_omega, min_sigma, max_sigma, L=2000):
    """calculate_igso3 pre-computes numerical approximations to the IGSO3 cdfs
    and score norms and expected squared score norms.

    Args:
        num_sigma: number of different sigmas for which to compute igso3
            quantities.
        num_omega: number of point in the discretization in the angle of
            rotation.
        min_sigma, max_sigma: the upper and lower ranges for the angle of
            rotation on which to consider the IGSO3 distribution.  This cannot
            be too low or it will create numerical instability.
    """
    # Discretize omegas for calculating CDFs. Skip omega=0.
    discrete_omega = np.linspace(0, np.pi, num_omega+1)[1:]

    # Exponential noise schedule.  This choice is closely tied to the
    # scalings used when simulating the reverse time SDE. For each step n,
    # discrete_sigma[n] = min_eps^(1-n/num_eps) * max_eps^(n/num_eps)
    discrete_sigma = 10 ** np.linspace(np.log10(min_sigma), np.log10(max_sigma), num_sigma + 1)[1:]

    # Compute the pdf and cdf values for the marginal distribution of the angle
    # of rotation (which is needed for sampling)
    pdf_vals = np.asarray(
        [igso3_density_angle(discrete_omega, sigma**2, L=L) for sigma in discrete_sigma])
    cdf_vals = np.asarray(
        [pdf.cumsum() / num_omega * np.pi for pdf in pdf_vals])

    # Compute the norms of the scores.  This are used to scale the rotation axis when
    # computing the score as a vector.
    score_norm = np.asarray(
        [d_logf_d_omega(discrete_omega, sigma**2, L=L) for sigma in discrete_sigma])

    # Compute the standard deviation of the score norm for each sigma
    exp_score_norms = np.sqrt(
        np.sum(
            score_norm**2 * pdf_vals, axis=1) / np.sum(
                pdf_vals, axis=1))
    return {
        'cdf': cdf_vals,
        'score_norm': score_norm,
        'exp_score_norms': exp_score_norms,
        'discrete_omega': discrete_omega,
        'discrete_sigma': discrete_sigma,
    }

def write_pkl(save_path: str, pkl_data):
    """Serialize data into a pickle file."""
    with open(save_path, "wb") as handle:
        pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pkl(read_path: str, verbose=False):
    """Read data from a pickle file."""
    with open(read_path, "rb") as handle:
        try:
            return pickle.load(handle)
        except Exception as e:
            if verbose:
                print(f"Failed to read {read_path}")
            raise (e)

class IGSO3:
    """
    Class for taking in a set of backbone crds and performing IGSO3 diffusion
    on all of them.

    Unlike the diffusion on translations, much of this class is written for a
    scaling between an initial time t=0 and final time t=1.
    """

    def __init__(
        self,
        *,
        T,
        min_sigma,
        max_sigma,
        min_b,
        max_b,
        cache_dir,
        num_omega=1000,
        schedule="linear",
        L=2000,
    ):
        """

        Args:
            T: total number of time steps
            min_sigma: smallest allowed scale parameter, should be at least 0.01 to maintain numerical stability.  Recommended value is 0.05.
            max_sigma: for exponential schedule, the largest scale parameter. Ignored for recommeded linear schedule
            min_b: lower value of beta in Ho schedule analogue
            max_b: upper value of beta in Ho schedule analogue
            num_omega: discretization level in the angles across [0, pi]
            schedule: currently only linear and exponential are supported.  The exponential schedule may be noising too slowly.
            L: truncation level
        """
        self._log = logging.getLogger(__name__)

        self.T = T

        self.schedule = schedule
        self.cache_dir = cache_dir
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

        if self.schedule == "linear":
            self.min_b = min_b
            self.max_b = max_b
            self.max_sigma = self.sigma(1.0)
        self.num_omega = num_omega
        self.num_sigma = 500
        # Calculate igso3 values.
        self.L = L  # truncation level
        self.igso3_vals = self._calc_igso3_vals(L=L)
        self.step_size = 1 / self.T

    def _calc_igso3_vals(self, L=2000):
        """_calc_igso3_vals computes numerical approximations to the
        relevant analytically intractable functionals of the igso3
        distribution.

        The calculated values are cached, or loaded from cache if they already
        exist.

        Args:
            L: truncation level for power series expansion of the pdf.
        """
        replace_period = lambda x: str(x).replace(".", "_")
        if self.schedule == "linear":
            cache_fname = os.path.join(
                self.cache_dir,
                f"T_{self.T}_omega_{self.num_omega}_min_sigma_{replace_period(self.min_sigma)}"
                + f"_min_b_{replace_period(self.min_b)}_max_b_{replace_period(self.max_b)}_schedule_{self.schedule}.pkl",
            )
        elif self.schedule == "exponential":
            cache_fname = os.path.join(
                self.cache_dir,
                f"T_{self.T}_omega_{self.num_omega}_min_sigma_{replace_period(self.min_sigma)}"
                f"_max_sigma_{replace_period(self.max_sigma)}_schedule_{self.schedule}",
            )
        else:
            raise ValueError(f"Unrecognize schedule {self.schedule}")

        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        if os.path.exists(cache_fname):
            self._log.info("Using cached IGSO3.")
            igso3_vals = read_pkl(cache_fname)
        else:
            self._log.info("Calculating IGSO3.")
            igso3_vals = calculate_igso3(
                num_sigma=self.num_sigma,
                min_sigma=self.min_sigma,
                max_sigma=self.max_sigma,
                num_omega=self.num_omega,
                L=L
            )
            write_pkl(cache_fname, igso3_vals)

        return igso3_vals

    @property
    def discrete_sigma(self):
        return self.igso3_vals["discrete_sigma"]

    def sigma_idx(self, sigma: np.ndarray):
        """
        Calculates the index for discretized sigma during IGSO(3) initialization."""
        return np.digitize(sigma, self.discrete_sigma) - 1

    def t_to_idx(self, t: np.ndarray):
        """
        Helper function to go from discrete time index t to corresponding sigma_idx.

        Args:
            t: time index (integer between 1 and 200)
        """
        continuous_t = t / self.T
        return self.sigma_idx(self.sigma(continuous_t))

    def sigma(self, t: torch.tensor):
        """
        Extract \sigma(t) corresponding to chosen sigma schedule.

        Args:
            t: torch tensor with time between 0 and 1
        """
        if not type(t) == torch.Tensor:
            t = torch.tensor(t)
        if torch.any(t < 0) or torch.any(t > 1):
            raise ValueError(f"Invalid t={t}")
        if self.schedule == "exponential":
            sigma = t * np.log10(self.max_sigma) + (1 - t) * np.log10(self.min_sigma)
            return 10**sigma
        elif self.schedule == "linear":  # Variance exploding analogue of Ho schedule
            # add self.min_sigma for stability
            return (
                self.min_sigma
                + t * self.min_b
                + (1 / 2) * (t**2) * (self.max_b - self.min_b)
            )
        else:
            raise ValueError(f"Unrecognize schedule {self.schedule}")

    def g(self, t):
        """
        g returns the drift coefficient at time t

        since
            sigma(t)^2 := \int_0^t g(s)^2 ds,
        for arbitrary sigma(t) we invert this relationship to compute
            g(t) = sqrt(d/dt sigma(t)^2).

        Args:
            t: scalar time between 0 and 1

        Returns:
            drift cooeficient as a scalar.
        """
        t = torch.tensor(t, requires_grad=True)
        sigma_sqr = self.sigma(t) ** 2
        grads = torch.autograd.grad(sigma_sqr.sum(), t)[0]
        return torch.sqrt(grads)

    def sample(self, ts, n_samples=1):
        """
        sample uses the inverse cdf to sample an angle of rotation from
        IGSO(3)

        Args:
            ts: array of integer time steps to sample from.
            n_samples: number of samples to draw.
        Returns:
        sampled angles of rotation. [len(ts), N]
        """
        assert sum(ts == 0) == 0, "assumes one-indexed, not zero indexed"
        all_samples = []
        for t in ts:
            sigma_idx = self.t_to_idx(t)
            sample_i = np.interp(
                np.random.rand(n_samples),
                self.igso3_vals["cdf"][sigma_idx],
                self.igso3_vals["discrete_omega"],
            )  # [N, 1]
            all_samples.append(sample_i)
        return np.stack(all_samples, axis=0)

    def sample_vec(self, ts, n_samples=1):  # same as diffdock
        """sample_vec generates a rotation vector(s) from IGSO(3) at time steps
        ts.

        Return:
            Sampled vector of shape [len(ts), N, 3]
        """
        x = np.random.randn(len(ts), n_samples, 3)
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
        return x * self.sample(ts, n_samples=n_samples)[..., None]

    def score_norm(self, t, omega):
        """
        score_norm computes the score norm based on the time step and angle
        Args:
            t: integer time step
            omega: angles (scalar or shape [N])
        Return:
            score_norm with same shape as omega
        """
        sigma_idx = self.t_to_idx(t)
        score_norm_t = np.interp(
            omega,
            self.igso3_vals["discrete_omega"],
            self.igso3_vals["score_norm"][sigma_idx],
        )
        return score_norm_t

    def score_vec(self, ts, vec):
        """score_vec computes the score of the IGSO(3) density as a rotation
        vector. This score vector is in the direction of the sampled vector,
        and has magnitude given by score_norms.

        In particular, Rt @ hat(score_vec(ts, vec)) is what is referred to as
        the score approximation in Algorithm 1


        Args:
            ts: times of shape [T]
            vec: where to compute the score of shape [T, N, 3]
        Returns:
            score vectors of shape [T, N, 3]
        """
        omega = np.linalg.norm(vec, axis=-1)
        all_score_norm = []
        for i, t in enumerate(ts):
            omega_t = omega[i]
            t_idx = t - 1
            sigma_idx = self.t_to_idx(t)
            score_norm_t = np.interp(
                omega_t,
                self.igso3_vals["discrete_omega"],
                self.igso3_vals["score_norm"][sigma_idx],
            )[:, None]
            all_score_norm.append(score_norm_t)
        score_norm = np.stack(all_score_norm, axis=0)
        return score_norm * vec / omega[..., None]

    def exp_score_norm(self, ts):
        """exp_score_norm returns the expected value of norm of the score for
        IGSO(3) with time parameter ts of shape [T].
        """
        sigma_idcs = [self.t_to_idx(t) for t in ts]
        return self.igso3_vals["exp_score_norms"][sigma_idcs]

    def diffuse_frames(self, xyz, t_list, diffusion_mask=None):
        """diffuse_frames samples from the IGSO(3) distribution to noise frames

        Parameters:
            xyz (np.array or torch.tensor, required): (L,3,3) set of backbone coordinates
            mask (np.array or torch.tensor, required): (L,) set of bools. True/1 is NOT diffused, False/0 IS diffused
        Returns:
            np.array : N/CA/C coordinates for each residue
                        (T,L,3,3), where T is num timesteps
        """

        if torch.is_tensor(xyz):
            xyz = xyz.numpy()

        t = np.arange(self.T) + 1  # 1-indexed!!
        num_res = len(xyz)

        N = torch.from_numpy(xyz[None, :, 0, :])
        Ca = torch.from_numpy(xyz[None, :, 1, :])  # [1, num_res, 3, 3]
        C = torch.from_numpy(xyz[None, :, 2, :])

        # scipy rotation object for true coordinates
        # Supplementary eq(1): for a given residue we may apply a Gram-Schmidt process to compute a 3 × 3
        # rotation matrix r with rows
        # with C_alpha at the origin
        R_true, Ca = rigid_from_3_points(N, Ca, C)
        R_true = R_true[0]
        Ca = Ca[0]

        # Sample rotations and scores from IGSO3
        sampled_rots = self.sample_vec(t, n_samples=num_res)  # [T, N, 3] eq(4) in RFDiffusion Supplementary

        if diffusion_mask is not None:
            non_diffusion_mask = 1 - diffusion_mask[None, :, None]
            sampled_rots = sampled_rots * non_diffusion_mask

        # Apply sampled rot.
        R_sampled = (
            scipy_R.from_rotvec(sampled_rots.reshape(-1, 3))
            .as_matrix()
            .reshape(self.T, num_res, 3, 3)
        )
        R_perturbed = np.einsum("tnij,njk->tnik", R_sampled, R_true)
        perturbed_crds = (
            np.einsum(
                "tnij,naj->tnai", R_sampled, xyz[:, :3, :] - Ca[:, None, ...].numpy()
            )
            + Ca[None, :, None].numpy()
        )

        if t_list != None:
            idx = [i - 1 for i in t_list]
            perturbed_crds = perturbed_crds[idx]
            R_perturbed = R_perturbed[idx]

        return (
            perturbed_crds.transpose(1, 0, 2, 3),  # [L, T, 3, 3]  # coordinates
            R_perturbed.transpose(1, 0, 2, 3),
        )

    def diffuse_rots(self, x_0, timestep, diffusion_mask=None):
        """
        Parameters:
            x_0 [batch, num_res, 3, 3]
            timestep [batch]
        Returns:
            np.array : [batch, num_res, 3, 3]
        """
        batch, num_res = x_0.shape[:2]

        # Sample rotations and scores from IGSO3
        sampled_rots = self.sample_vec(timestep, n_samples=num_res)  # [batch, num_res, 3] eq(4) in RFDiffusion Supplementary       

        if diffusion_mask is not None:
            non_diffusion_mask = 1 - diffusion_mask[None, :, None]
            sampled_rots = sampled_rots * non_diffusion_mask
        # Apply sampled rot.
        R_sampled = (
            scipy_R.from_rotvec(sampled_rots.reshape(-1, 3))
            .as_matrix()
            .reshape(batch, num_res, 3, 3)
        )
        R_perturbed = np.einsum("tnij,tnjk->tnik", R_sampled, x_0)
        return R_perturbed

    def reverse_sample_vectorized(
        self, R_t, R_0, t, noise_level, mask=None, return_perturb=False
    ):
        """reverse_sample uses an approximation to the IGSO3 score to sample
        a rotation at the previous time step.

        Roughly - this update follows the reverse time SDE for Reimannian
        manifolds proposed by de Bortoli et al. Theorem 1 [1]. But with an
        approximation to the score based on the prediction of R0.
        Unlike in reference [1], this diffusion on SO(3) relies on geometric
        variance schedule.  Specifically we follow [2] (appendix C) and assume
            sigma_t = sigma_min * (sigma_max / sigma_min)^{t/T},
        for time step t.  When we view this as a discretization  of the SDE
        from time 0 to 1 with step size (1/T).  Following Eq. 5 and Eq. 6,
        this maps on to the forward  time SDEs
            dx = g(t) dBt [FORWARD]
        and
            dx = g(t)^2 score(xt, t)dt + g(t) B't, [REVERSE]
        where g(t) = sigma_t * sqrt(2 * log(sigma_max/ sigma_min)), and Bt and
        B't are Brownian motions. The formula for g(t) obtains from equation 9
        of [2], from which this sampling function may be generalized to
        alternative noising schedules.
        Args:
            R_t: noisy rotation of shape [N, 3, 3]
            R_0: prediction of un-noised rotation
            t: integer time step
            noise_level: scaling on the noise added when obtaining sample
                (preliminary performance seems empirically better with noise
                level=0.5)
            mask: whether the residue is to be updated.  A value of 1 means the
                rotation is not updated from r_t.  A value of 0 means the
                rotation is updated.
        Return:
            sampled rotation matrix for time t-1 of shape [3, 3]
        Reference:
        [1] De Bortoli, V., Mathieu, E., Hutchinson, M., Thornton, J., Teh, Y.
        W., & Doucet, A. (2022). Riemannian score-based generative modeling.
        arXiv preprint arXiv:2202.02763.
        [2] Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S.,
        & Poole, B. (2020). Score-based generative modeling through stochastic
        differential equations. arXiv preprint arXiv:2011.13456.
        """
        # compute rotation vector corresponding to prediction of how r_t goes to r_0
        R_0, R_t = torch.tensor(R_0), torch.tensor(R_t)
        R_0t = torch.einsum("...ij,...kj->...ik", R_t, R_0)
        R_0t_rotvec = torch.tensor(
            scipy_R.from_matrix(R_0t.cpu().numpy()).as_rotvec()
        ).to(R_0.device)

        # Approximate the score based on the prediction of R0.
        # R_t @ hat(Score_approx) is the score approximation in the Lie algebra
        # SO(3) (i.e. the output of Algorithm 1)
        Omega = torch.linalg.norm(R_0t_rotvec, axis=-1).numpy()
        Score_approx = R_0t_rotvec * (self.score_norm(t, Omega) / Omega)[:, None]

        # Compute scaling for score and sampled noise (following Eq 6 of [2])
        continuous_t = t / self.T
        rot_g = self.g(continuous_t).to(Score_approx.device)

        # Sample and scale noise to add to the rotation perturbation in the
        # SO(3) tangent space.  Since IG-SO(3) is the Brownian motion on SO(3)
        # (up to a deceleration of time by a factor of two), for small enough
        # time-steps, this is equivalent to perturbing r_t with IG-SO(3) noise.
        # See e.g. Algorithm 1 of De Bortoli et al.
        Z = np.random.normal(size=(R_0.shape[0], 3))
        Z = torch.from_numpy(Z).to(Score_approx.device)
        Z *= noise_level

        Delta_r = (rot_g**2) * self.step_size * Score_approx

        # Sample perturbation from discretized SDE (following eq. 6 of [2]),
        # This approximate sampling from IGSO3(* ; Delta_r, rot_g^2 *
        # self.step_size) with tangent Gaussian.
        Perturb_tangent = Delta_r + rot_g * np.sqrt(self.step_size) * Z
        if mask is not None:
            Perturb_tangent *= (1 - mask.long())[:, None, None]
        Perturb = Exp(Perturb_tangent)

        if return_perturb:
            return Perturb

        Interp_rot = torch.einsum("...ij,...jk->...ik", Perturb, R_t)

        return Interp_rot


# More complicated version splits error in CA-N and CA-C (giving more accurate CB position)
# It returns the rigid transformation from local frame to global frame
def rigid_from_3_points(N, Ca, C, non_ideal=False, eps=1e-8):  # AF2 Supplementary Algorithm 21
    # N, Ca, C - [B,L, 3]
    # R - [B,L, 3, 3], det(R)=1, inv(R) = R.T, R is a rotation matrix
    B, L = N.shape[:2]
    # eq (1) in RFDiffusion supplementary
    v1 = C - Ca
    v2 = N - Ca
    e1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + eps)  # r1
    u2 = v2 - (torch.einsum("bli, bli -> bl", e1, v2)[..., None] * e1)
    e2 = u2 / (torch.norm(u2, dim=-1, keepdim=True) + eps)  # r2
    e3 = torch.cross(e1, e2, dim=-1)  # r3
    R = torch.cat(
        [e1[..., None], e2[..., None], e3[..., None]], axis=-1
    )  # [B,L,3,3] - rotation matrix

    return R, Ca

if __name__ == '__main__':
    rots = np.array([  # (1,4,3,3)
        [[ # identity
            [1,0,0],
            [0,1,0],
            [0,0,1]
        ],
        [
            [0,0,1],
            [0,1.,0],
            [-1,0,0]
        ],  
        [
            [0,0,-1],
            [0,1.,0],
            [1,0,0]
        ],
        [
            [0,0,1],
            [0,-1.,0],
            [1,0,0]
        ]],
        ], dtype=np.float32)
    igso3 = IGSO3(
            T=1000,
            min_sigma=0.02,
            max_sigma=1.5,
            schedule="linear",
            min_b=1.5,
            max_b=2.5,
            cache_dir="log/igso3",
            L=2000,
        )
    print(igso3.diffuse_rots(rots,np.array([1])))
    print(igso3.diffuse_rots(rots,np.array([100])))
    print(igso3.diffuse_rots(rots,np.array([500])))