import paddle
import paddle.nn as nn
import tools.lddt as lddt
from tools import quat_affine, residue_constants
from tools.model_utils import init_final_linear
import numpy as np
from tools import r3, all_atom


def generate_new_affine(sequence_mask):
    t_shape = sequence_mask.shape[:-1] # (batch, N_res, 1)
    assert len(t_shape) == 2
    t_shape.append(3) # (batch, N_res, 3)
    q_shape = sequence_mask.shape[:-1] + [1] # (batch, N_res, 1)
    quaternion = paddle.tile(
                    paddle.reshape(
                        paddle.to_tensor([1.0, 0.0, 0.0, 0.0]), [1, 1, 4]),
                    repeat_times=q_shape)
    translation = paddle.zeros(t_shape)
    return quat_affine.QuatAffine(quaternion, translation)


def sigmoid_cross_entropy(logits, labels):
    """Computes sigmoid cross entropy given logits and multiple class labels."""
    log_p = nn.functional.log_sigmoid(logits)
    # log(1 - sigmoid(x)) = log_sigmoid(-x), the latter is more numerically stable
    log_not_p = nn.functional.log_sigmoid(-logits)
    loss = -labels * log_p - (1. - labels) * log_not_p
    return loss


def softmax_cross_entropy(logits, labels):
    """Computes softmax cross entropy given logits and one-hot class labels."""
    loss = -paddle.sum(labels * nn.functional.log_softmax(logits), axis=-1)
    return loss


def _distogram_log_loss(logits, bin_edges, batch, num_bins):
    """Log loss of a distogram."""
    positions = batch['pseudo_beta']
    mask = batch['pseudo_beta_mask']

    assert positions.shape[-1] == 3

    sq_breaks = paddle.square(bin_edges).unsqueeze([1, 2])

    dist2 = paddle.sum(
        paddle.square(
            paddle.unsqueeze(positions, axis=-2) -
            paddle.unsqueeze(positions, axis=-3)),
        axis=-1,
        keepdim=True)

    true_bins = paddle.sum(dist2 > sq_breaks, axis=-1)

    errors = softmax_cross_entropy(
        labels=nn.functional.one_hot(true_bins, num_classes=num_bins), logits=logits)

    square_mask = paddle.unsqueeze(mask, axis=-2) * paddle.unsqueeze(mask, axis=-1)

    avg_error = (
        paddle.sum(errors * square_mask, axis=[-2, -1]) /
        (1e-6 + paddle.sum(square_mask, axis=[-2, -1])))
    dist2 = dist2[..., 0]
    return {
        'loss': avg_error, 
        'true_dist': paddle.sqrt(1e-6 + dist2)}


def l2_normalize(x, axis=-1, epsilon=1e-12):
    return x / paddle.sqrt(
        paddle.maximum(
            paddle.sum(paddle.square(x), axis=axis, keepdim=True),
            paddle.to_tensor([epsilon], dtype='float32')))


def squared_difference(x, y):
    return paddle.square(x - y)


class MaskedMsaHead(nn.Layer):
    """Head to predict MSA at the masked locations.

    The MaskedMsaHead employs a BERT-style objective to reconstruct a masked
    version of the full MSA, based on a linear projection of
    the MSA representation.
    Jumper et al. (2021) Suppl. Sec. 1.9.9 "Masked MSA prediction"
    """
    def __init__(self, channel_num, config, global_config, name='masked_msa_head'):
        super(MaskedMsaHead, self).__init__()
        self.config = config
        self.global_config = global_config
        self.num_output = config.num_output
        self.logits = nn.Linear(channel_num['msa_channel'], self.num_output, name='logits')

    def forward(self, msa_representation):
        """Builds MaskedMsaHead module.

        Arguments:
        representations: Dictionary of representations, must contain:
            * 'msa': MSA representation, shape [batch, N_seq, N_res, c_m].
        batch: Batch, unused.

        Returns:
        Dictionary containing:
            * 'logits': logits of shape [batch, N_seq, N_res, N_aatype] with
                (unnormalized) log probabilies of predicted aatype at position.
        """
        logits = self.logits(msa_representation['msa'])
        return {logits:logits}

    def loss(self, value, batch):
        errors = softmax_cross_entropy(
            labels=nn.functional.one_hot(batch['true_msa'], num_classes=self.num_output),
            logits=value['logits'])
        loss = (paddle.sum(errors * batch['bert_mask'], axis=[-2, -1]) /
                (1e-8 + paddle.sum(batch['bert_mask'], axis=[-2, -1])))
        return {'loss': loss}


class PredictedLDDTHead(nn.Layer):
    """Head to predict the per-residue LDDT to be used as a confidence measure.

    Jumper et al. (2021) Suppl. Sec. 1.9.6 "Model confidence prediction (pLDDT)"
    Jumper et al. (2021) Suppl. Alg. 29 "predictPerResidueLDDT_Ca"
    """

    def __init__(self, channel_num, config, global_config, name='predicted_lddt_head'):
        super(PredictedLDDTHead, self).__init__()
        self.config = config
        self.global_config = global_config

        self.input_layer_norm = nn.LayerNorm(channel_num['seq_channel'],
                                             name='input_layer_norm')
        self.act_0 = nn.Linear(channel_num['seq_channel'],
                               self.config.num_channels, name='act_0')
        self.act_1 = nn.Linear(self.config.num_channels,
                               self.config.num_channels, name='act_1')
        self.logits = nn.Linear(self.config.num_channels,
                               self.config.num_bins, name='logits')

    def forward(self, representations):
        """Builds PredictedLDDTHead module.

        Arguments:
        representations: Dictionary of representations, must contain:
            * 'structure_module': Single representation from the structure module,
                shape [n_batch, N_res, c_s].

        Returns:
        Dictionary containing :
            * 'logits': logits of shape [n_batch, N_res, N_bins] with
                (unnormalized) log probabilies of binned predicted lDDT.
        """
        act = representations['structure_module']
        act = self.input_layer_norm(act)
        act = nn.functional.relu(self.act_0(act))
        act = nn.functional.relu(self.act_1(act))
        logits = self.logits(act)

        return dict(logits=logits)

    def loss(self, value, batch):
        # Shape (n_batch, num_res, 37, 3)
        pred_all_atom_pos = value['structure_module']['final_atom_positions']
        # Shape (n_batch, num_res, 37, 3)
        true_all_atom_pos = paddle.cast(batch['all_atom_positions'], 'float32')
        # Shape (n_batch, num_res, 37)
        all_atom_mask = paddle.cast(batch['all_atom_mask'], 'float32')

        # Shape (batch_size, num_res)
        lddt_ca = lddt.lddt(
            # Shape (batch_size, num_res, 3)
            predicted_points=pred_all_atom_pos[:, :, 1, :],
            # Shape (batch_size, num_res, 3)
            true_points=true_all_atom_pos[:, :, 1, :],
            # Shape (batch_size, num_res, 1)
            true_points_mask=all_atom_mask[:, :, 1:2],
            cutoff=15.,
            per_residue=True)
        lddt_ca = lddt_ca.detach()

        # Shape (batch_size, num_res)
        num_bins = self.config.num_bins
        bin_index = paddle.floor(lddt_ca * num_bins)

        # protect against out of range for lddt_ca == 1
        bin_index = paddle.minimum(bin_index, paddle.to_tensor(num_bins - 1, dtype='float32'))
        lddt_ca_one_hot = nn.functional.one_hot(paddle.cast(bin_index, 'int64'), num_classes=num_bins)

        # Shape (n_batch, num_res, num_channel)
        logits = value['predicted_lddt']['logits']
        errors = softmax_cross_entropy(labels=lddt_ca_one_hot, logits=logits)

        # Shape (num_res,)
        mask_ca = all_atom_mask[:, :, residue_constants.atom_order['CA']]
        mask_ca = paddle.to_tensor(mask_ca, dtype='float32')
        loss = paddle.sum(errors * mask_ca, axis=-1) / (paddle.sum(mask_ca, axis=-1) + 1e-8)

        if self.config.filter_by_resolution:
            # NMR & distillation have resolution = 0
            resolution = paddle.squeeze(batch['resolution'], axis=-1)
            loss *= paddle.cast((resolution >= self.config.min_resolution)
                    & (resolution <= self.config.max_resolution), 'float32')
        output = {'loss': loss}
        return output


class PredictedAlignedErrorHead(nn.Layer):
    """Head to predict the distance errors in the backbone alignment frames.

    Can be used to compute predicted TM-Score.
    Jumper et al. (2021) Suppl. Sec. 1.9.7 "TM-score prediction"
    """
    def __init__(self, channel_num, config, global_config,
                 name='predicted_aligned_error_head'):
        super(PredictedAlignedErrorHead, self).__init__()
        self.config = config
        self.global_config = global_config

        self.logits = nn.Linear(channel_num['pair_channel'],
                                self.config.num_bins, name='logits')

    def forward(self, representations):
        """Builds PredictedAlignedErrorHead module.

        Arguments:
            representations: Dictionary of representations, must contain:
                * 'pair': pair representation, shape [B, N_res, N_res, c_z].
            batch: Batch, unused.

        Returns:
            Dictionary containing:
                * logits: logits for aligned error, shape [B, N_res, N_res, N_bins].
                * bin_breaks: array containing bin breaks, shape [N_bins - 1].
        """
        logits = self.logits(representations['pair'])
        breaks = paddle.linspace(0., self.config.max_error_bin,
                                 self.config.num_bins-1)

        return dict(logits=logits, breaks=breaks)

    def loss(self, value, batch):
        # Shape (B, num_res, 7)
        predicted_affine = quat_affine.QuatAffine.from_tensor(
            value['structure_module']['final_affines'])
        # Shape (B, num_res, 7)
        true_rot = paddle.to_tensor(batch['backbone_affine_tensor_rot'], dtype='float32')
        true_trans = paddle.to_tensor(batch['backbone_affine_tensor_trans'], dtype='float32')
        true_affine = quat_affine.QuatAffine(
            quaternion=None,
            translation=true_trans,
            rotation=true_rot)
        # Shape (B, num_res)
        mask = batch['backbone_affine_mask']
        # Shape (B, num_res, num_res)
        square_mask = mask[..., None] * mask[:, None, :]
        num_bins = self.config.num_bins
        # (num_bins - 1)
        breaks = value['predicted_aligned_error']['breaks']
        # (B, num_res, num_res, num_bins)
        logits = value['predicted_aligned_error']['logits']

        # Compute the squared error for each alignment.
        def _local_frame_points(affine):
            points = [paddle.unsqueeze(x, axis=-2) for x in 
                            paddle.unstack(affine.translation, axis=-1)]
            return affine.invert_point(points, extra_dims=1)
        error_dist2_xyz = [
            paddle.square(a - b)
            for a, b in zip(_local_frame_points(predicted_affine),
                            _local_frame_points(true_affine))]
        error_dist2 = sum(error_dist2_xyz)
        # Shape (B, num_res, num_res)
        # First num_res are alignment frames, second num_res are the residues.
        error_dist2 = error_dist2.detach()

        sq_breaks = paddle.square(breaks)
        true_bins = paddle.sum(paddle.cast((error_dist2[..., None] > sq_breaks), 'int32'), axis=-1)

        errors = softmax_cross_entropy(
            labels=paddle.nn.functional.one_hot(true_bins, num_classes=num_bins), logits=logits)

        loss = (paddle.sum(errors * square_mask, axis=[-2, -1]) /
            (1e-8 + paddle.sum(square_mask, axis=[-2, -1])))

        if self.config.filter_by_resolution:
            # NMR & distillation have resolution = 0
            resolution = paddle.squeeze(batch['resolution'], axis=-1)
            loss *= paddle.cast((resolution >= self.config.min_resolution)
                    & (resolution <= self.config.max_resolution), 'float32')

        output = {'loss': loss}
        return output


class ExperimentallyResolvedHead(nn.Layer):
    """Predicts if an atom is experimentally resolved in a high-res structure.

    Only trained on high-resolution X-ray crystals & cryo-EM.
    Jumper et al. (2021) Suppl. Sec. 1.9.10 '"Experimentally resolved" prediction'
    """

    def __init__(self, channel_num, config, global_config, name='experimentally_resolved_head'):
        super(ExperimentallyResolvedHead, self).__init__()
        self.config = config
        self.global_config = global_config
        self.logits = nn.Linear(channel_num['seq_channel'], 37, name='logits')

    def forward(self, representations):
        """Builds ExperimentallyResolvedHead module.

        Arguments:
        representations: Dictionary of representations, must contain:
            * 'single': Single representation, shape [B, N_res, c_s].
        batch: Batch, unused.

        Returns:
        Dictionary containing:
            * 'logits': logits of shape [B, N_res, 37],
                log probability that an atom is resolved in atom37 representation,
                can be converted to probability by applying sigmoid.
        """
        logits = self.logits(representations['single'])
        return dict(logits=logits)

    def loss(self, value, batch):
        logits = value['logits']
        assert len(logits.shape) == 3

        # Does the atom appear in the amino acid?
        atom_exists = batch['atom37_atom_exists']
        # Is the atom resolved in the experiment? Subset of atom_exists,
        # *except for OXT*
        all_atom_mask = paddle.cast(batch['all_atom_mask'], 'float32')

        xent = sigmoid_cross_entropy(labels=all_atom_mask, logits=logits)
        loss = paddle.sum(xent * atom_exists, axis=[-2, -1]) / (1e-8 + paddle.sum(atom_exists, axis=[-2, -1]))

        if self.config.filter_by_resolution:
            # NMR & distillation have resolution = 0
            resolution = paddle.squeeze(batch['resolution'], axis=-1)
            loss *= paddle.cast((resolution >= self.config.min_resolution)
                    & (resolution <= self.config.max_resolution), 'float32')

        output = {'loss': loss}
        return output


class DistogramHead(nn.Layer):
    """Head to predict a distogram.

    Jumper et al. (2021) Suppl. Sec. 1.9.8 "Distogram prediction"
    """

    def __init__(self, channel_num, config, name='distogram_head'):
        super(DistogramHead, self).__init__()
        self.config = config
        # self.global_config = global_config

        self.half_logits = nn.Linear(channel_num['pair_channel'],
                                    self.config.num_bins, name='half_logits')
        init_final_linear(self.half_logits)

    def forward(self, representations):
        """Builds DistogramHead module.

        Arguments:
        representations: Dictionary of representations, must contain:
            * 'pair': pair representation, shape [batch, N_res, N_res, c_z].

        Returns:
        Dictionary containing:
            * logits: logits for distogram, shape [batch, N_res, N_res, N_bins].
            * bin_breaks: array containing bin breaks, shape [batch, N_bins - 1].
        """
        half_logits = self.half_logits(representations['pair'])

        logits = half_logits + paddle.transpose(half_logits, perm=[0, 2, 1, 3])
        breaks = paddle.linspace(self.config.first_break, self.config.last_break,
                          self.config.num_bins - 1)
        breaks = paddle.tile(breaks[None, :],
                            repeat_times=[logits.shape[0], 1])

        return {
            'logits': logits, 
            'bin_edges': breaks}

    def loss(self, value, batch):
        return _distogram_log_loss(value['logits'], value['bin_edges'],
                               batch, self.config.num_bins)


class InvariantPointAttention(nn.Layer):
    """Invariant Point attention module.

    The high-level idea is that this attention module works over a set of points
    and associated orientations in 3D space (e.g. protein residues).

    Each residue outputs a set of queries and keys as points in their local
    reference frame.  The attention is then defined as the euclidean distance
    between the queries and keys in the global frame.

    Jumper et al. (2021) Suppl. Alg. 22 "InvariantPointAttention"
    """
    def __init__(self, channel_num, config, global_config,
                 dist_epsilon=1e-8):
        super(InvariantPointAttention, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config
        self.dist_epsilon = dist_epsilon

        num_head = self.config.num_head
        num_scalar_qk = self.config.num_scalar_qk
        num_point_qk = self.config.num_point_qk
        num_scalar_v = self.config.num_scalar_v
        num_point_v = self.config.num_point_v
        num_output = self.config.num_channel

        assert num_scalar_qk > 0
        assert num_point_qk > 0
        assert num_point_v > 0

        self.q_scalar = nn.Linear(
            channel_num['seq_channel'], num_head * num_scalar_qk)
        self.kv_scalar = nn.Linear(
            channel_num['seq_channel'],
            num_head * (num_scalar_v + num_scalar_qk))

        self.q_point_local = nn.Linear(
            channel_num['seq_channel'], num_head * 3 * num_point_qk)
        self.kv_point_local = nn.Linear(
            channel_num['seq_channel'],
            num_head * 3 * (num_point_qk + num_point_v))

        tpw = np.log(np.exp(1.) - 1.)
        self.trainable_point_weights = paddle.create_parameter(
            [num_head], 'float32',
            default_initializer=nn.initializer.Constant(tpw))

        self.attention_2d = nn.Linear(channel_num['pair_channel'], num_head)

        if self.global_config.zero_init:
            init_w = nn.initializer.Constant(value=0.0)
        else:
            init_w = nn.initializer.XavierUniform()

        c = num_scalar_v + num_point_v * 4 + channel_num['pair_channel']
        self.output_projection = nn.Linear(
            num_head * c, num_output,
            weight_attr=paddle.ParamAttr(initializer=init_w))

    def forward(self, single_act: paddle.Tensor, pair_act: paddle.Tensor,
                mask: paddle.Tensor, affine: quat_affine.QuatAffine):
        # single_act: [B, N, C]
        # pair_act: [B, N, M, C']
        # mask: [B, N, 1]
        num_residues = single_act.shape[1]
        num_head = self.config.num_head
        num_scalar_qk = self.config.num_scalar_qk
        num_point_qk = self.config.num_point_qk
        num_scalar_v = self.config.num_scalar_v
        num_point_v = self.config.num_point_v
        num_output = self.config.num_channel

        # Construct scalar queries of shape:
        # [batch_size, num_query_residues, num_head, num_points]
        q_scalar = self.q_scalar(single_act)
        q_scalar = paddle.reshape(
            q_scalar, [-1, num_residues, num_head, num_scalar_qk])

        # Construct scalar keys/values of shape:
        # [batch_size, num_target_residues, num_head, num_points]
        kv_scalar = self.kv_scalar(single_act)
        kv_scalar = paddle.reshape(
            kv_scalar,
            [-1, num_residues, num_head, num_scalar_v + num_scalar_qk])
        k_scalar, v_scalar = paddle.split(
            kv_scalar, [num_scalar_qk, -1], axis=-1)

        # Construct query points of shape:
        # [batch_size, num_residues, num_head, num_point_qk]
        q_point_local = self.q_point_local(single_act)
        q_point_local = paddle.split(q_point_local, 3, axis=-1)

        q_point_global = affine.apply_to_point(q_point_local, extra_dims=1)
        q_point = [
            paddle.reshape(x, [-1, num_residues, num_head, num_point_qk])
            for x in q_point_global]

        # Construct key and value points.
        # Key points shape [batch_size, num_residues, num_head, num_point_qk]
        # Value points shape [batch_size, num_residues, num_head, num_point_v]
        kv_point_local = self.kv_point_local(single_act)
        kv_point_local = paddle.split(kv_point_local, 3, axis=-1)

        kv_point_global = affine.apply_to_point(kv_point_local, extra_dims=1)
        kv_point_global = [
            paddle.reshape(x, [-1, num_residues, num_head, num_point_qk + num_point_v])
            for x in kv_point_global]

        k_point, v_point = list(
            zip(*[
                paddle.split(x, [num_point_qk, -1], axis=-1)
                for x in kv_point_global
            ]))

        # We assume that all queries and keys come iid from N(0, 1) distribution
        # and compute the variances of the attention logits.
        # Each scalar pair (q, k) contributes Var q*k = 1
        scalar_variance = max(num_scalar_qk, 1) * 1.
        # Each point pair (q, k) contributes Var [0.5 ||q||^2 - <q, k>] = 9 / 2
        point_variance = max(num_point_qk, 1) * 9. / 2

        # Allocate equal variance to scalar, point and attention 2d parts so that
        # the sum is 1.

        num_logit_terms = 3
        scalar_weights = np.sqrt(1.0 / (num_logit_terms * scalar_variance))
        point_weights = np.sqrt(1.0 / (num_logit_terms * point_variance))
        attention_2d_weights = np.sqrt(1.0 / (num_logit_terms))

        trainable_point_weights = nn.functional.softplus(
            self.trainable_point_weights)
        point_weights *= paddle.unsqueeze(
            trainable_point_weights, axis=1)

        # [B, R, H, C] => [B, H, R, C], put head dim first
        q_point = [paddle.transpose(x, [0, 2, 1, 3]) for x in q_point]
        k_point = [paddle.transpose(x, [0, 2, 1, 3]) for x in k_point]
        v_point = [paddle.transpose(x, [0, 2, 1, 3]) for x in v_point]

        dist2 = [
            paddle.square(paddle.unsqueeze(qx, axis=-2) - \
                          paddle.unsqueeze(kx, axis=-3))
            for qx, kx in zip(q_point, k_point)]
        dist2 = sum(dist2)

        attn_qk_point = -0.5 * paddle.sum(
            paddle.unsqueeze(point_weights, axis=[1, 2]) * dist2, axis=-1)

        q = paddle.transpose(scalar_weights * q_scalar, [0, 2, 1, 3])
        k = paddle.transpose(k_scalar, [0, 2, 1, 3])
        v = paddle.transpose(v_scalar, [0, 2, 1, 3])
        attn_qk_scalar = paddle.matmul(q, paddle.transpose(k, [0, 1, 3, 2]))
        attn_logits = attn_qk_scalar + attn_qk_point

        attention_2d = self.attention_2d(pair_act)
        attention_2d = paddle.transpose(attention_2d, [0, 3, 1, 2])
        attention_2d = attention_2d_weights * attention_2d
        attn_logits += attention_2d

        mask_2d = mask * paddle.transpose(mask, [0, 2, 1])
        attn_logits -= 1e5 * (1. - mask_2d.unsqueeze(1))

        # [batch_size, num_head, num_query_residues, num_target_residues]
        attn = nn.functional.softmax(attn_logits)

        # o_i^h
        # [batch_size, num_query_residues, num_head, num_head * num_scalar_v]
        result_scalar = paddle.matmul(attn, v)
        result_scalar = paddle.transpose(result_scalar, [0, 2, 1, 3])

        # o_i^{hp}
        # [batch_size, num_query_residues, num_head, num_head * num_point_v]
        result_point_global = [
            paddle.sum(paddle.unsqueeze(attn, -1) * paddle.unsqueeze(vx, -3),
                       axis=-2) for vx in v_point]
        result_point_global = [
            paddle.transpose(x, [0, 2, 1, 3]) for x in result_point_global]

        # \tilde{o}_i^h
        # [batch_size, num_residues, num_head, pair_channel]
        result_attention_over_2d = paddle.einsum(
            'nhij,nijc->nihc', attn, pair_act)

        # Reshape, global-to-local and save
        result_scalar = paddle.reshape(
            result_scalar, [-1, num_residues, num_head * num_scalar_v])
        result_point_global = [
            paddle.reshape(x, [-1, num_residues, num_head * num_point_v])
            for x in result_point_global]
        result_point_local = affine.invert_point(
            result_point_global, extra_dims=1)
        result_attention_over_2d = paddle.reshape(
            result_attention_over_2d,
            [-1, num_residues, num_head * self.channel_num['pair_channel']])

        result_point_local_norm = paddle.sqrt(
            self.dist_epsilon + paddle.square(result_point_local[0]) + \
            paddle.square(result_point_local[1]) + \
            paddle.square(result_point_local[2]))

        output_features = [result_scalar]
        output_features.extend(result_point_local)
        output_features.extend(
            [result_point_local_norm, result_attention_over_2d])

        final_act = paddle.concat(output_features, axis=-1)
        return self.output_projection(final_act)


class MultiRigidSidechain(nn.Layer):
    """Class to make side chain atoms."""
    def __init__(self, channel_num, config, global_config):
        super(MultiRigidSidechain, self).__init__()

        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        c = self.config.num_channel
        self.input_projection = nn.Linear(channel_num['seq_channel'], c)
        self.input_projection_1 = nn.Linear(channel_num['seq_channel'], c)

        for i in range(self.config.num_residual_block):
            l1, l2 = 'resblock1', 'resblock2'
            if i > 0:
                l1, l2 = f'resblock1_{i}', f'resblock2_{i}'

            init_w_1 = nn.initializer.KaimingNormal()
            if self.global_config.zero_init:
                init_w_2 = nn.initializer.Constant(value=0.)
            else:
                init_w_2 = nn.initializer.XavierUniform()

            setattr(self, l1, nn.Linear(
                c, c, weight_attr=paddle.ParamAttr(initializer=init_w_1)))
            setattr(self, l2, nn.Linear(
                c, c, weight_attr=paddle.ParamAttr(initializer=init_w_2)))

        self.unnormalized_angles = nn.Linear(c, 14)

    def forward(self, affine, single_act, init_single_act, aatype):
        single_act = self.input_projection(nn.functional.relu(single_act))
        init_single_act = self.input_projection_1(
            nn.functional.relu(init_single_act))
        act = single_act + init_single_act

        for i in range(self.config.num_residual_block):
            l1, l2 = 'resblock1', 'resblock2'
            if i > 0:
                l1, l2 = f'resblock1_{i}', f'resblock2_{i}'

            old_act = act
            act = getattr(self, l1)(nn.functional.relu(act))
            act = getattr(self, l2)(nn.functional.relu(act))
            act += old_act

        # Map activations to torsion angles. Shape: (num_res, 14).
        num_res = act.shape[1]
        unnormalized_angles = self.unnormalized_angles(
            nn.functional.relu(act))
        unnormalized_angles = paddle.reshape(
            unnormalized_angles, [-1, num_res, 7, 2])
        angles = l2_normalize(unnormalized_angles, axis=-1)

        outputs = {
            'angles_sin_cos': angles,  #  (B, N, 7, 2)
            'unnormalized_angles_sin_cos':
                unnormalized_angles,   #  (B, N, 7, 2)
        }

        # Jumper et al. (2021) Suppl. Alg. 24 "computeAllAtomCoordinates"
        backbone_to_global = r3.rigids_from_quataffine(affine)
        all_frames_to_global = all_atom.torsion_angles_to_frames(
            aatype, backbone_to_global, angles)
        pred_positions = all_atom.frames_and_literature_positions_to_atom14_pos(
            aatype, all_frames_to_global)

        # Outputs1 (Rot + Trans)
        outputs.update({
            'atom_pos': pred_positions.translation,  # (B, N, 14, 3)
            'frames_rot': all_frames_to_global.rot.rotation,  # (B, N, 8, 3, 3)
            'frames_trans': all_frames_to_global.trans.translation,  # (B, N, 8, 3)
        })

        # ## Outputs2 (Rigids)
        # outputs.update({
        #     'atom_pos': pred_positions.translation,  # (B, N, 14, 3)
        #     'frames': all_frames_to_global,  # (B, N, 8, 3, 3)
        # })

        return outputs


class FoldIteration(nn.Layer):
    """A single iteration of the main structure module loop.

    Jumper et al. (2021) Suppl. Alg. 20 "StructureModule" lines 6-21

    First, each residue attends to all residues using InvariantPointAttention.
    Then, we apply transition layers to update the hidden representations.
    Finally, we use the hidden representations to produce an update to the
    affine of each residue.
    """
    def __init__(self, channel_num, config, global_config):
        super(FoldIteration, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        self.invariant_point_attention = InvariantPointAttention(
            channel_num, config, global_config)
        self.attention_layer_norm = nn.LayerNorm(channel_num['seq_channel'])

        for i in range(self.config.num_layer_in_transition):
            if i < self.config.num_layer_in_transition - 1:
                init_w = nn.initializer.KaimingNormal()
            elif self.global_config.zero_init:
                init_w = nn.initializer.Constant(value=0.0)
            else:
                init_w = nn.initializer.XavierUniform()

            layer_name, c_in = 'transition', channel_num['seq_channel']
            if i > 0:
                layer_name, c_in = f'transition_{i}', self.config.num_channel

            setattr(self, layer_name, nn.Linear(
                c_in, self.config.num_channel,
                weight_attr=paddle.ParamAttr(initializer=init_w)))

        self.ipa_dropout = nn.Dropout(p=self.config.dropout)
        self.transition_dropout = nn.Dropout(p=self.config.dropout)
        self.transition_layer_norm = nn.LayerNorm(self.config.num_channel)

        if self.global_config.zero_init:
            last_init_w = nn.initializer.Constant(value=0.0)
        else:
            last_init_w = nn.initializer.XavierUniform()

        # Jumper et al. (2021) Alg. 23 "Backbone update"
        self.affine_update = nn.Linear(
            self.config.num_channel, 6,
            weight_attr=paddle.ParamAttr(initializer=last_init_w))

        self.rigid_sidechain = MultiRigidSidechain(
            channel_num, self.config.sidechain, self.global_config)

    def forward(self, activations, init_single_act, static_pair_act,
                seq_mask, aatype):
        affine = quat_affine.QuatAffine.from_tensor(activations['affine'])
        act = activations['act']

        attn = self.invariant_point_attention(
            act, static_pair_act, seq_mask, affine)
        act += attn
        act = self.ipa_dropout(act)
        act = self.attention_layer_norm(act)

        input_act = act
        for i in range(self.config.num_layer_in_transition):
            layer_name = 'transition'
            if i > 0:
                layer_name = f'transition_{i}'

            act = getattr(self, layer_name)(act)

            if i < self.config.num_layer_in_transition - 1:
                act = nn.functional.relu(act)

        act += input_act
        act = self.transition_dropout(act)
        act = self.transition_layer_norm(act)

        affine_update = self.affine_update(act)
        affine = affine.pre_compose(affine_update)

        sc = self.rigid_sidechain(
            affine.scale_translation(self.config.position_scale),
            act, init_single_act, aatype)
        outputs = {'affine': affine.to_tensor(), 'sc': sc}

        affine = affine.stop_rot_gradient()
        new_activations = {
            'act': act,
            'affine': affine.to_tensor()
        }
        return new_activations, outputs


class StructureModule(nn.Layer):
    """StructureModule as a network head.

    Jumper et al. (2021) Suppl. Alg. 20 "StructureModule"
    """
    def __init__(self, channel_num, config, global_config):
        super(StructureModule, self).__init__()
        assert config.num_layer > 0

        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        self.single_layer_norm = nn.LayerNorm(channel_num['seq_channel'])
        self.initial_projection = nn.Linear(
            channel_num['seq_channel'], config.num_channel)
        self.pair_layer_norm = nn.LayerNorm(channel_num['pair_channel'])

        self.fold_iteration = FoldIteration(
            channel_num, config, global_config)

    def forward(self, representations, batch):
        """tbd."""

        output = self._generate_affines(representations, batch)

        ret = dict()
        ret['representations'] = {'structure_module': output['act']}

        # NOTE: pred unit is nanometer, *position_scale to scale back to
        # angstroms to match unit of PDB files.
        # (L, B, N, 7), L = FoldIteration layers
        scale = paddle.to_tensor(
            [1.] * 4 + [self.config.position_scale] * 3, 'float32')
        ret['traj'] = output['affine'] * paddle.unsqueeze(
            scale, axis=[0, 1, 2])

        ret['sidechains'] = output['sc']

        # (B, N, 14, 3)
        atom14_pred_positions = output['sc']['atom_pos'][-1]
        ret['final_atom14_positions'] = atom14_pred_positions

        # (B, N, 14)
        ret['final_atom14_mask'] = batch['atom14_atom_exists']

        # (B, N, 37, 3)
        atom37_pred_positions = all_atom.atom14_to_atom37(
            atom14_pred_positions, batch)
        atom37_pred_positions *= paddle.unsqueeze(
            batch['atom37_atom_exists'], axis=-1)
        ret['final_atom_positions'] = atom37_pred_positions

        # (B, N, 37)
        ret['final_atom_mask'] = batch['atom37_atom_exists']

        # (B, N, 7)
        ret['final_affines'] = ret['traj'][-1]

        return ret

    def _generate_affines(self, representations, batch):
        """Generate predicted affines for a single chain.

        Jumper et al. (2021) Suppl. Alg. 20 "StructureModule"

        This is the main part of the structure module - it iteratively applies
        folding to produce a set of predicted residue positions.

        Args:
            representations: Representations dictionary.
            batch: Batch dictionary.

        Returns:
            A dictionary containing residue affines and sidechain positions.
        """
        seq_mask = paddle.unsqueeze(batch['seq_mask'], axis=-1)

        single_act = self.single_layer_norm(representations['single'])

        init_single_act = single_act
        single_act = self.initial_projection(single_act)
        pair_act = self.pair_layer_norm(representations['pair'])
        affine = generate_new_affine(seq_mask)

        outputs = []
        activations = {'act': single_act, 'affine': affine.to_tensor()}
        for _ in range(self.config.num_layer):
            activations, output = self.fold_iteration(
                activations, init_single_act, pair_act,
                seq_mask, batch['aatype'])
            outputs.append(output)

        output = dict()
        for k in outputs[0].keys():
            if k == 'sc':
                output[k] = dict()
                for l in outputs[0][k].keys():
                    output[k][l] = paddle.stack([o[k][l] for o in outputs])
            else:
                output[k] = paddle.stack([o[k] for o in outputs])

        output['act'] = activations['act']
        return output
