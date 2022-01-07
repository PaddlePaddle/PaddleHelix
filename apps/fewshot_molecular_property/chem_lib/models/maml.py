
import paddle.nn as nn
import paddlefsl.utils as utils

class MAML(nn.Layer):

	def __init__(
			self,
			model,
			lr,
			first_order=False,
			allow_unused=None,
			allow_nograd=False,
			anil=False,
	):
		super(MAML, self).__init__()
		self.layers = model
		self.lr = lr
		self.first_order = first_order
		self.allow_nograd = allow_nograd
		if allow_unused is None:
			allow_unused = allow_nograd
		self.allow_unused = allow_unused
		self.anil = anil

	def forward(self, *args, **kwargs):
		return self.layers(*args, **kwargs)

	def clone(self, first_order=None, allow_unused=None, allow_nograd=None,anil=None):
		"""
		**Description**

		Returns a `MAML`-wrapped copy of the module whose parameters and buffers
		are `torch.clone`d from the original module.

		This implies that back-propagating losses on the cloned module will
		populate the buffers of the original module.
		For more information, refer to learn2learn.clone_module().

		**Arguments**

		* **first_order** (bool, *optional*, default=None) - Whether the clone uses first-
			or second-order updates. Defaults to self.first_order.
		* **allow_unused** (bool, *optional*, default=None) - Whether to allow differentiation
		of unused parameters. Defaults to self.allow_unused.
		* **allow_nograd** (bool, *optional*, default=False) - Whether to allow adaptation with
			parameters that have `requires_grad = False`. Defaults to self.allow_nograd.

		"""
		if anil is None:
			anil = self.anil
		if first_order is None:
			first_order = self.first_order
		if allow_unused is None:
			allow_unused = self.allow_unused
		if allow_nograd is None:
			allow_nograd = self.allow_nograd
		return MAML(
			utils.clone_model(self.layers),
			lr=self.lr,
			first_order=first_order,
			allow_unused=allow_unused,
			allow_nograd=allow_nograd,
			anil=anil
		)
