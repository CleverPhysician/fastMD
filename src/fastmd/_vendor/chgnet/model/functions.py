from __future__ import annotations

import itertools
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from fastmd._vendor.chgnet.model.triton_fusions import model_fusions_enabled


def aggregate(data: Tensor, owners: Tensor, *, average=True, num_owner=None) -> Tensor:
    """Aggregate rows in data by specifying the owners.

    Args:
        data (Tensor): data tensor to aggregate [n_row, feature_dim]
        owners (Tensor): specify the owner of each row [n_row, 1]
        average (bool): if True, average the rows, if False, sum the rows.
            Default = True
        num_owner (int, optional): the number of owners, this is needed if the
            max idx of owner is not presented in owners tensor
            Default = None

    Returns:
        output (Tensor): [num_owner, feature_dim]
    """
    if num_owner is None:
        bin_count = torch.bincount(owners)
        num_owner = bin_count.shape[0]
    elif average:
        bin_count = torch.bincount(owners, minlength=num_owner)[:num_owner]

    # A caller-provided num_owner makes the output shape static. In particular,
    # sum aggregation can then avoid a dynamic-output bincount during CUDA Graph
    # capture.
    output = data.new_zeros([num_owner, data.shape[1]])
    output = output.index_add_(0, owners, data)
    if average:
        bin_count = bin_count.where(bin_count != 0, bin_count.new_ones(1))
        output = (output.T / bin_count).T
    return output


class MLP(nn.Module):
    """Multi-Layer Perceptron used for non-linear regression."""

    def __init__(
        self,
        input_dim: int,
        *,
        output_dim: int = 1,
        hidden_dim: int | Sequence[int] | None = (64, 64),
        dropout: float = 0,
        activation: str = "silu",
        bias: bool = True,
    ) -> None:
        """Initialize the MLP.

        Args:
            input_dim (int): the input dimension
            output_dim (int): the output dimension
            hidden_dim (list[int] | int]): a list of integers or a single integer
                representing the number of hidden units in each layer of the MLP.
                Default = [64, 64]
            dropout (float): the dropout rate before each linear layer. Default: 0
            activation (str, optional): The name of the activation function to use
                in the gated MLP. Must be one of "relu", "silu", "tanh", or "gelu".
                Default = "silu"
            bias (bool): whether to use bias in each Linear layers.
                Default = True
        """
        super().__init__()
        if hidden_dim is None or hidden_dim == 0:
            layers = [nn.Dropout(dropout), nn.Linear(input_dim, output_dim, bias=bias)]
        elif isinstance(hidden_dim, int):
            layers = [
                nn.Linear(input_dim, hidden_dim, bias=bias),
                find_activation(activation),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim, bias=bias),
            ]
        elif isinstance(hidden_dim, Sequence):
            layers = [
                nn.Linear(input_dim, hidden_dim[0], bias=bias),
                find_activation(activation),
            ]
            if len(hidden_dim) != 1:
                for h_in, h_out in itertools.pairwise(hidden_dim):
                    layers.append(nn.Linear(h_in, h_out, bias=bias))
                    layers.append(find_activation(activation))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim[-1], output_dim, bias=bias))
        else:
            raise TypeError(
                f"{hidden_dim=} must be an integer, a list of integers, or None."
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Performs a forward pass through the MLP.

        Args:
            x (Tensor): a tensor of shape (batch_size, input_dim)

        Returns:
            Tensor: a tensor of shape (batch_size, output_dim)
        """
        return self.layers(x)


class GatedMLP(nn.Module):
    """Gated MLP
    similar model structure is used in CGCNN and M3GNet.
    """

    # A batched second projection only amortizes its activation/repack kernel
    # for the largest triplet tensors (notably the 512-atom MgO workloads).
    _fused_bmm_min_rows = 100_000

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int | list[int] | None = None,
        dropout: float = 0,
        activation: str = "silu",
        norm: str = "batch",
        bias: bool = True,
    ) -> None:
        """Initialize a gated MLP.

        Args:
            input_dim (int): the input dimension
            output_dim (int): the output dimension
            hidden_dim (list[int] | int]): a list of integers or a single integer
                representing the number of hidden units in each layer of the MLP.
                Default = None
            dropout (float): the dropout rate before each linear layer.
                Default: 0
            activation (str, optional): The name of the activation function to use in
                the gated MLP. Must be one of "relu", "silu", "tanh", or "gelu".
                Default = "silu"
            norm (str, optional): The name of the normalization layer to use on the
                updated atom features. Must be one of "batch", "layer", or None.
                Default = "batch"
            bias (bool): whether to use bias in each Linear layers.
                Default = True
        """
        super().__init__()
        self.mlp_core = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation=activation,
            bias=bias,
        )
        self.mlp_gate = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation=activation,
            bias=bias,
        )
        self.activation = find_activation(activation)
        self.sigmoid = nn.Sigmoid()
        self.norm = norm
        self.bn1 = find_normalization(name=norm, dim=output_dim)
        self.bn2 = find_normalization(name=norm, dim=output_dim)
        self.register_buffer("_fused_input_weight", None, persistent=False)
        self.register_buffer("_fused_input_bias", None, persistent=False)
        self.register_buffer("_fused_output_weight", None, persistent=False)
        self.register_buffer("_fused_output_bias", None, persistent=False)
        self._fused_depth = 0
        self._fused_source_signature: tuple | None = None

    @staticmethod
    def _linear_layers(mlp: MLP) -> list[nn.Linear]:
        """Return the ordered Linear layers in one GatedMLP branch."""
        return [module for module in mlp.layers if isinstance(module, nn.Linear)]

    @staticmethod
    def _supported_branch(mlp: MLP, depth: int) -> bool:
        """Check the exact frozen CHGNet branch patterns optimized below."""
        modules = list(mlp.layers)
        if depth == 1:
            return (
                len(modules) == 2
                and isinstance(modules[0], nn.Dropout)
                and modules[0].p == 0
                and isinstance(modules[1], nn.Linear)
            )
        if depth == 2:
            return (
                len(modules) == 4
                and isinstance(modules[0], nn.Linear)
                and isinstance(modules[1], nn.SiLU)
                and isinstance(modules[2], nn.Dropout)
                and modules[2].p == 0
                and isinstance(modules[3], nn.Linear)
            )
        return False

    @staticmethod
    def _source_signature(linears: list[nn.Linear]) -> tuple:
        """Identify packed source tensors and detect later in-place updates."""
        tensors = [
            tensor
            for linear in linears
            for tensor in (linear.weight, linear.bias)
            if tensor is not None
        ]
        return tuple(
            (
                id(tensor),
                tensor._version,
                tensor.data_ptr(),
                tensor.device,
                tensor.dtype,
                tensor.requires_grad,
                tuple(tensor.shape),
            )
            for tensor in tensors
        )

    def prepare_fused_inference(self) -> bool:
        """Prepack the frozen core/gate input projection for CUDA MD inference."""
        self._fused_depth = 0
        self._fused_source_signature = None
        self._fused_input_weight = None
        self._fused_input_bias = None
        self._fused_output_weight = None
        self._fused_output_bias = None
        if self.training or self.norm != "layer":
            return False
        if not (
            isinstance(self.activation, nn.SiLU)
            and isinstance(self.sigmoid, nn.Sigmoid)
            and isinstance(self.bn1, nn.LayerNorm)
            and isinstance(self.bn2, nn.LayerNorm)
            and self.bn1.elementwise_affine
            and self.bn2.elementwise_affine
            and self.bn1.eps == self.bn2.eps
        ):
            return False
        parameters = tuple(self.parameters())
        if any(parameter.requires_grad for parameter in parameters):
            return False
        core_linears = self._linear_layers(self.mlp_core)
        gate_linears = self._linear_layers(self.mlp_gate)
        depth = len(core_linears)
        if depth not in {1, 2} or len(gate_linears) != depth:
            return False
        if not (
            self._supported_branch(self.mlp_core, depth)
            and self._supported_branch(self.mlp_gate, depth)
        ):
            return False
        for core_linear, gate_linear in zip(core_linears, gate_linears, strict=True):
            if (
                core_linear.weight.shape != gate_linear.weight.shape
                or (core_linear.bias is None) != (gate_linear.bias is None)
            ):
                return False
        if any(
            linear.bias is None for linear in (*core_linears, *gate_linears)
        ):
            return False
        first_core, first_gate = core_linears[0], gate_linears[0]
        if (
            not first_core.weight.is_cuda
            or first_core.weight.dtype != torch.float32
            or first_core.bias is None
            or first_gate.bias is None
        ):
            return False
        self._fused_input_weight = torch.cat(
            (first_core.weight, first_gate.weight), dim=0
        ).detach()
        self._fused_input_bias = torch.cat(
            (first_core.bias, first_gate.bias), dim=0
        ).detach()
        if depth == 2:
            second_core, second_gate = core_linears[1], gate_linears[1]
            self._fused_output_weight = torch.stack(
                (second_core.weight, second_gate.weight), dim=0
            ).detach()
            self._fused_output_bias = torch.stack(
                (second_core.bias, second_gate.bias), dim=0
            ).detach()[:, None, :]
        sources = [*core_linears, *gate_linears]
        self._fused_source_signature = self._source_signature(sources)
        self._fused_depth = depth
        return True

    def _fused_inference_ready(self, x: Tensor) -> bool:
        """Validate that a prepacked frozen projection remains current."""
        core_linears = self._linear_layers(self.mlp_core)
        gate_linears = self._linear_layers(self.mlp_gate)
        if (
            self._fused_depth not in {1, 2}
            or self._fused_input_weight is None
            or self._fused_input_bias is None
            or (
                self._fused_depth == 2
                and (
                    self._fused_output_weight is None
                    or self._fused_output_bias is None
                )
            )
            or self.training
            or any(parameter.requires_grad for parameter in self.parameters())
            or not x.is_cuda
            or x.dtype != torch.float32
            or x.device != self._fused_input_weight.device
            or self.norm != "layer"
            or not isinstance(self.activation, nn.SiLU)
            or not isinstance(self.sigmoid, nn.Sigmoid)
            or not isinstance(self.bn1, nn.LayerNorm)
            or not isinstance(self.bn2, nn.LayerNorm)
            or not self.bn1.elementwise_affine
            or not self.bn2.elementwise_affine
            or self.bn1.eps != self.bn2.eps
            or len(core_linears) != self._fused_depth
            or len(gate_linears) != self._fused_depth
            or not self._supported_branch(self.mlp_core, self._fused_depth)
            or not self._supported_branch(self.mlp_gate, self._fused_depth)
            or any(
                linear.bias is None
                for linear in (*core_linears, *gate_linears)
            )
        ):
            return False
        sources = [*core_linears, *gate_linears]
        return self._fused_source_signature == self._source_signature(sources)

    def _forward_fused_inference(self, x: Tensor) -> Tensor:
        """Run packed projections and a fused frozen LayerNorm/gating epilogue."""
        from fastmd._vendor.chgnet.model.ops import layer_norm_silu_gate

        packed = F.linear(x, self._fused_input_weight, self._fused_input_bias)
        if self._fused_depth == 2:
            if packed.shape[0] >= self._fused_bmm_min_rows:
                from fastmd._vendor.chgnet.model.ops import silu_repack

                hidden = silu_repack(packed)
                projected = torch.baddbmm(
                    self._fused_output_bias,
                    hidden,
                    self._fused_output_weight.transpose(1, 2),
                )
                core, gate = projected.unbind(0)
            else:
                packed = F.silu(packed)
                core_hidden, gate_hidden = packed.chunk(2, dim=-1)
                core_linear = self._linear_layers(self.mlp_core)[1]
                gate_linear = self._linear_layers(self.mlp_gate)[1]
                core = F.linear(core_hidden, core_linear.weight, core_linear.bias)
                gate = F.linear(gate_hidden, gate_linear.weight, gate_linear.bias)
        else:
            core, gate = packed.chunk(2, dim=-1)
        return layer_norm_silu_gate(
            core,
            gate,
            self.bn1.weight,
            self.bn1.bias,
            self.bn2.weight,
            self.bn2.bias,
            self.bn1.eps,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Performs a forward pass through the MLP.

        Args:
            x (Tensor): a tensor of shape (batch_size, input_dim)

        Returns:
            Tensor: a tensor of shape (batch_size, output_dim)
        """
        if (
            model_fusions_enabled()
            and self._fused_inference_ready(x)
        ):
            return self._forward_fused_inference(x)
        if self.norm is None:
            core = self.activation(self.mlp_core(x))
            gate = self.sigmoid(self.mlp_gate(x))
        else:
            core = self.activation(self.bn1(self.mlp_core(x)))
            gate = self.sigmoid(self.bn2(self.mlp_gate(x)))
        return core * gate


class ScaledSiLU(torch.nn.Module):
    """Scaled Sigmoid Linear Unit."""

    def __init__(self) -> None:
        """Initialize a scaled SiLU."""
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = torch.nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self._activation(x) * self.scale_factor


def find_activation(name: str) -> nn.Module:
    """Return an activation function using name."""
    try:
        return {
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "scaledsilu": ScaledSiLU,
            "gelu": nn.GELU,
            "softplus": nn.Softplus,
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh,
        }[name.lower()]()
    except KeyError as exc:
        raise NotImplementedError from exc


def find_normalization(name: str, dim: int | None = None) -> nn.Module | None:
    """Return an normalization function using name."""
    if name is None:
        return None
    return {
        "batch": nn.BatchNorm1d(dim),
        "layer": nn.LayerNorm(dim),
    }.get(name.lower(), None)
