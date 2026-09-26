from __future__ import annotations

from collections.abc import Sequence
import os

import torch
from torch import Tensor, nn
import math
from .op import fused_silu, fused_sigmoid

# The crossover is workload/GPU dependent and can be tuned without changing
# model code.  Below the threshold PyTorch's LayerNorm kernels are faster.
_LN_FUSE_THRESH = int(os.getenv("MATRIS_GATED_LN_FUSE_MIN_ROWS", "10000"))
_SEGMENT_SOFTMAX_FUSION_ENABLED = os.getenv(
    "MATRIS_FUSED_SEGMENT_SOFTMAX", "1"
) != "0"
_GATED_LN_FUSION_ENABLED = os.getenv("MATRIS_FUSED_GATED_LN", "1") != "0"
_FUSED_GATED_LN_RESIDUAL = os.getenv(
    "MATRIS_FUSED_GATED_LN_RESIDUAL", "1"
) != "0"
_MODEL_FUSIONS_ENABLED = os.getenv("MATRIS_MODEL_FUSIONS", "1") != "0"
_ACT_SILU = 1
_ACT_SIGMOID = 2


def set_model_fusions_enabled(enabled: bool) -> bool:
    """Set the process-local generic fusion switch and return its old value."""
    global _MODEL_FUSIONS_ENABLED
    previous = _MODEL_FUSIONS_ENABLED
    _MODEL_FUSIONS_ENABLED = bool(enabled)
    return previous


def model_fusions_enabled() -> bool:
    return _MODEL_FUSIONS_ENABLED

class FusedSiLU(torch.nn.Module):
    """Fused Sigmoid Linear Unit."""

    def __init__(self) -> None:
        """Initialize a fused SiLU."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        if x.device.type == "cuda":
            return fused_silu(x)
        else:
            return torch.nn.functional.silu(x) 

class FusedSigmoid(torch.nn.Module):
    """Fused Sigmoid Linear Unit."""

    def __init__(self) -> None:
        """Initialize a fused SiLU."""
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        if x.device.type == "cuda":
            return fused_sigmoid(x)
        else:
            return torch.nn.functional.sigmoid(x)

def get_activation(name: str) -> nn.Module:
    """Return an activation function"""
    activation_map = {
        "relu": nn.ReLU,
        "silu": FusedSiLU,  # Using fused version for better performance
        "gelu": nn.GELU,
        "softplus": nn.Softplus,
        "sigmoid": FusedSigmoid,  # Using fused version for better performance
        "tanh": nn.Tanh,
    }
    
    name_lower = name.lower()
    if name_lower not in activation_map:
        raise NotImplementedError(
            f"Activation '{name}' is not implemented. "
            f"Supported activations: {list(activation_map.keys())}"
        )
    return activation_map[name_lower]()

def get_normalization(name: str, dim: int | None = None) -> nn.Module | None:
    """Return an normalization function"""
    if name is None:
        return None
        
    normalization_map = {
        "layer": nn.LayerNorm(dim),
        "rms": nn.RMSNorm(dim), # torch >= 2.6.0
        "batch": nn.BatchNorm1d(dim),
    }
    name_lower = name.lower()
    return normalization_map[name_lower]

class SwishLayer(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        output_dim: int = 128,
        bias: bool = True,
    ) -> None:
        """
        Args:
            input_dim: Input dimension.
            output_dim: Output dimension.
            bias: Whether to use bias in the linear layer. Default: True.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.act = get_activation("silu")
    
    def forward(self, feas: Tensor) -> Tensor:
        """
        Args:
            feas: shape (feas_num, in_dim)
            
        Returns:
            output: shape (feas_num, out_dim)
        """
        return self.act(self.linear(feas))

def segment_count(segment: Tensor, num_segment: int) -> Tensor:
    """Sync-free replacement for ``torch.bincount(segment, minlength=num_segment)``.

    ``torch.bincount`` reads ``max()`` on the host (a D2H sync / stall) on CUDA
    even when ``minlength`` is given. Counting via ``index_add_`` into a
    fixed-size buffer is sync-free and CUDA-graph capturable.
    """
    counts = torch.zeros(num_segment, dtype=torch.long, device=segment.device)
    counts.index_add_(0, segment.long(), torch.ones_like(segment, dtype=torch.long))
    return counts


def Dimwise_softmax(feas: Tensor, segment: Tensor, num_segment=None) -> Tensor:
    """Computes a sparsely evaluated softmax.
    
    Args:
        feas: The source tensor. shape: [num, dim]
        segment: specify the segment of each row [num, 1] 
    """
    num, dim = feas.shape
    if num_segment is None:
        num_segment = int(segment.max()) + 1
    if (
        _MODEL_FUSIONS_ENABLED
        and _SEGMENT_SOFTMAX_FUSION_ENABLED
        and feas.is_cuda
    ):
        from .op.triton_segment_softmax import fused_segment_softmax

        return fused_segment_softmax(feas, segment, num_segment)
    
    segment_expanded = segment.unsqueeze(1).expand(-1, dim) # [num, dim]
    
    feas_max = torch.empty( num_segment, dim, dtype=feas.dtype, device=feas.device )
    feas_max.fill_(float("-inf"))
    feas_max = feas_max.scatter_reduce(
        0, segment_expanded, feas, reduce='amax', include_self=False,
    ) #[num_segment, dim]
    # Gather: [num_segment, dim] -> [num, dim]
    feas_max = feas_max[segment]
    out = (feas - feas_max).exp()
    
    # =========== scatter sum ============
    out_sum = torch.zeros(num_segment, dim, device=feas.device, dtype=feas.dtype)
    out_sum = out_sum.scatter_reduce(
        0, segment_expanded, out, reduce='sum', include_self=False
    )
    # Gather: [num_segment, dim] -> [num, dim]
    out_sum = out_sum[segment]
    score = out / out_sum
    return score

def aggregate(data: torch.Tensor, 
              segment: torch.Tensor, 
              bin_count: torch.Tensor = None, 
              average=True, 
              num_segment=None) -> torch.Tensor:
    """Aggregate rows in data by specifying the segment.

    Args:
        data (Tensor): data tensor to aggregate [n_row, feature_dim]
        segment (Tensor): specify the owner of each row [n_row, 1]
        average (bool): if True, average the rows, if False, sum the rows.
            Default = True
        num_owner (int, optional): the number of owners, this is needed if the
            max idx of owner is not presented in owners tensor
            Default = None

    Returns:
        output (Tensor): [num_owner, feature_dim]
    """
    if bin_count is None:
        # torch.bincount syncs (reads max() on host) even with minlength; use a
        # sync-free index_add_ count when the segment size is known.
        if num_segment is not None:
            bin_count = segment_count(segment, num_segment)
        else:
            bin_count = torch.bincount(segment)
        bin_count = bin_count.where(bin_count != 0, bin_count.new_ones(1))

    if (num_segment is not None) and (bin_count.shape[0] != num_segment):
        difference = num_segment - bin_count.shape[0]
        bin_count = torch.cat([bin_count, bin_count.new_ones(difference)])
    # make sure this operation is done on the same device of data and owners
    output = data.new_zeros([bin_count.shape[0], data.shape[1]])
    output = output.index_add_(0, segment, data)
    if average:
        output = (output.T / bin_count).T
    return output


class MLP(nn.Module):
        
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int | Sequence[int] | None = (128, 128),
        output_dim: int = 128,
        dropout: float = 0.0,
        activation: Literal["silu", "relu", "tanh", "gelu"] = "silu",
        bias: bool = True,
        use_fp16: bool = False,
    ):
        """Initialize the MLP layer.
        Args:
            input_dim: Dimension of input features.
            hidden_dim: Number of hidden units. Can be an integer for a single
                hidden layer, a sequence of integers for multiple hidden layers,
                or None for no hidden layers. Default: (128, 128).
            output_dim: Dimension of output predictions. Default: 128.
            dropout: Dropout rate applied before each linear layer. Default: 0.0.
            activation: Activation function. Supported: "relu", "silu", "tanh", "gelu".
            bias: Whether to use bias in linear layers. Default: True.
            use_fp16: Whether to use mixed precision (FP16). Default: False.
        """
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"Dropout rate must be in [0.0, 1.0), got {dropout}")
        
        self.use_fp16 = use_fp16
        activation_func = get_activation(activation)

        layers = []
        if hidden_dim in (None, 0):
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(input_dim, output_dim, bias=bias))
        elif isinstance(hidden_dim, int):
            # Single hidden layer
            layers.extend([
                nn.Linear(input_dim, hidden_dim, bias=bias),
                activation_func,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim, bias=bias),
            ])
        elif isinstance(hidden_dim, Sequence):
            # Multiple hidden layers
            layers.extend([
                nn.Linear(input_dim, hidden_dim[0], bias=bias),
                activation_func,
            ])
            # Additional hidden layers
            for i in range(len(hidden_dim) - 1):
                layers.extend([
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim[i], hidden_dim[i + 1], bias=bias),
                    activation_func,
                ])
            # Output layer
            layers.extend([nn.Dropout(dropout), nn.Linear(hidden_dim[-1], output_dim, bias=bias)])
        else:
            raise TypeError(
                f"hidden_dim must be an integer, sequence of integers, or None, "
                f"got {type(hidden_dim).__name__}"
            )
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, feas: Tensor) -> Tensor:
        """
            Args:
                feas: Input tensor of shape (features, input_dim)
            Returns:
                Output tensor of shape (features, output_dim)
        """
        if self.use_fp16 and feas.is_cuda:
            with torch.amp.autocast(dtype=torch.float16, device_type="cuda"):
                out = self.layers(feas)
            out = out.to(torch.float32)
        else:
            out = self.layers(feas) 
        return out


class GatedMLP(nn.Module):
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int | Sequence[int] | None = (128, 128),
        output_dim: int = 128,
        dropout: float = 0.0,
        activation: str = "silu",
        norm_type: str = "layer",
        bias: bool = True,
        use_fp16: bool = False,
    ) -> None:
        """
        Args:
            input_dim: The input dimension.
            hidden_dim: A list of integers or a single integer representing the number 
                of hidden units in each layer of the MLP. Default: None.
            output_dim: The output dimension.
            dropout: The dropout rate. Default: 0.0.
            activation: The name of the activation function. Must be one of "relu", 
                "silu", "tanh", or "gelu". Default: "silu".
            norm_type: The name of the normalization layer to use. Must be one of 
                "layer", "rms", "batch", "group", or None. Default: "layer".
            bias: Whether to use bias in linear layers. Default: True.
            use_fp16: Whether to use mixed precision (FP16). Default: False.
        """
        super().__init__()
        self.use_fp16 = use_fp16
        self.activation_func = get_activation(activation)
        self.activation_gate = get_activation("sigmoid")
        self.gate_norm = get_normalization(name=norm_type, dim=output_dim)
        self.core_norm = get_normalization(name=norm_type, dim=output_dim)
        self.mlp_core = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation,
            bias=bias,
            use_fp16=use_fp16,
        )
        self.mlp_gate = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation,
            bias=bias,
            use_fp16=use_fp16,
        )

        # This fusion is generic: it only depends on LayerNorm and activation
        # types, rather than any MatRIS graph semantics.
        self._fuse_ln = (
            isinstance(self.core_norm, nn.LayerNorm)
            and isinstance(self.gate_norm, nn.LayerNorm)
            and isinstance(self.activation_func, FusedSiLU)
            and isinstance(self.activation_gate, FusedSigmoid)
        )
        # Keep Triton as a lazy CUDA-only dependency; these values mirror the
        # public activation constants in triton_layernorm_act.
        self._core_act, self._gate_act = _ACT_SILU, _ACT_SIGMOID

    def _finish_gate(self, core_x: Tensor, gate_x: Tensor) -> Tensor:
        if self.gate_norm is None:
            return self.activation_func(core_x) * self.activation_gate(gate_x)

        if (
            _MODEL_FUSIONS_ENABLED
            and _GATED_LN_FUSION_ENABLED
            and self._fuse_ln
            and core_x.is_cuda
            and core_x.shape[0] >= _LN_FUSE_THRESH
        ):
            core_norm = self.core_norm
            gate_norm = self.gate_norm
            if not (
                core_norm.weight.requires_grad
                or core_norm.bias.requires_grad
                or gate_norm.weight.requires_grad
                or gate_norm.bias.requires_grad
            ):
                from .op.triton_gated_ln import fused_gated_ln

                return fused_gated_ln(
                    core_x,
                    gate_x,
                    core_norm.weight,
                    core_norm.bias,
                    gate_norm.weight,
                    gate_norm.bias,
                    core_norm.eps,
                    gate_norm.eps,
                )

            from .op.triton_layernorm_act import fused_ln_act

            core = fused_ln_act(
                core_x,
                core_norm.weight,
                core_norm.bias,
                core_norm.eps,
                self._core_act,
            )
            gate = fused_ln_act(
                gate_x,
                gate_norm.weight,
                gate_norm.bias,
                gate_norm.eps,
                self._gate_act,
            )
            return core * gate

        core = self.activation_func(self.core_norm(core_x))
        gate = self.activation_gate(self.gate_norm(gate_x))
        return core * gate

    def forward_with_residual(
        self,
        feas: Tensor,
        residual: Tensor,
        res_weight: Tensor,
    ) -> Tensor | None:
        """Fuse the frozen GatedMLP epilogue with its weighted residual."""
        if not model_fusions_enabled() or not _FUSED_GATED_LN_RESIDUAL:
            return None
        if not (
            self._fuse_ln
            and feas.is_cuda
            and residual.is_cuda
            and res_weight.is_cuda
            and residual.dim() == 2
            and residual.shape[0] >= _LN_FUSE_THRESH
            and residual.shape[1] == res_weight.numel()
            and not res_weight.requires_grad
        ):
            return None
        core_norm = self.core_norm
        gate_norm = self.gate_norm
        if (
            core_norm.weight.requires_grad
            or core_norm.bias.requires_grad
            or gate_norm.weight.requires_grad
            or gate_norm.bias.requires_grad
        ):
            return None
        core_x = self.mlp_core(feas)
        gate_x = self.mlp_gate(feas)
        if core_x.shape != residual.shape or gate_x.shape != residual.shape:
            return None
        from .op.triton_gated_ln import fused_gated_ln_residual

        return fused_gated_ln_residual(
            core_x,
            gate_x,
            core_norm.weight,
            core_norm.bias,
            gate_norm.weight,
            gate_norm.bias,
            residual,
            res_weight,
            core_norm.eps,
            gate_norm.eps,
        )

    def forward(self, feas: Tensor) -> Tensor:
        """
        Args:
            feas (Tensor): shape (feas_num, input_dim)
        Returns:
            output: shape (feas_num, output_dim)
        """
        core_x = self.mlp_core(feas)
        gate_x = self.mlp_gate(feas)
        return self._finish_gate(core_x, gate_x)


class MOE_Layer(nn.Module):
    
    def __init__(
        self,
        num_expert: int = 64,
        input_dim: int = 128,
        hidden_dim: int | Sequence[int] | None = (128, 128),
        output_dim: int = 128,
        dropout: float = 0.0,
        activation: Literal["silu", "relu", "tanh", "gelu"] = "silu",
        bias: bool = True,
        use_fp16: bool = False,
    ):
        """Initialize the MOE layer.

        Args:
            
        """
        super().__init__()
        
        raise NotImplementedError
         
    def forward(self, feas: Tensor) -> Tensor:
        return None


class GraphPooling(nn.Module):
    def __init__(self, average: bool = False) -> None:
        
        super().__init__()
        self.average = average

    def forward(self, node_feat: Tensor, segment: Tensor, num_segment: int | None = None) -> Tensor:
        """
        Args:
            atom_feat (Tensor): batched atom features after convolution layers.
                [num_batch_atoms, node_feat_dim or 1]
            segment (Tensor): graph indices for each atom.
                [num_batch_atoms]
            num_segment (int): number of output segments (e.g. n_graphs). When
                provided, avoids a host max()-read (D2H sync).

        Returns:
            crystal_feas (Tensor): crystal feature matrix.
                [n_crystals, node_feat_dim or 1]
        """
        if self.average:
            if num_segment is not None:
                bin_count = segment_count(segment, num_segment)  # sync-free
            else:
                bin_count = torch.bincount(segment)
            bin_count = bin_count.where(bin_count != 0, bin_count.new_ones(1))
            n_seg = bin_count.shape[0]
        else:
            # average=False only needs the output size; avoid the host max()-read.
            n_seg = num_segment if num_segment is not None else int(segment.max()) + 1

        output = node_feat.new_zeros([n_seg, node_feat.shape[1]])
        output = output.index_add_(0, segment, node_feat)
        if self.average:
            output = (output.T / bin_count).T
        return output


def cg_change_mat(ang_mom: int, device: str = "cpu") -> torch.tensor:
    if ang_mom not in [2]:
        raise NotImplementedError

    if ang_mom == 2:
        change_mat = torch.tensor(
            [
                [3 ** (-0.5), 0, 0, 0, 3 ** (-0.5), 0, 0, 0, 3 ** (-0.5)],
                [0, 0, 0, 0, 0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0],
                [0, 0, -(2 ** (-0.5)), 0, 0, 0, 2 ** (-0.5), 0, 0],
                [0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0, 0, 0, 0, 0],
                [0, 0, 0.5**0.5, 0, 0, 0, 0.5**0.5, 0, 0],
                [0, 2 ** (-0.5), 0, 2 ** (-0.5), 0, 0, 0, 0, 0],
                [
                    -(6 ** (-0.5)),
                    0,
                    0,
                    0,
                    2 * 6 ** (-0.5),
                    0,
                    0,
                    0,
                    -(6 ** (-0.5)),
                ],
                [0, 0, 0, 0, 0, 2 ** (-0.5), 0, 2 ** (-0.5), 0],
                [-(2 ** (-0.5)), 0, 0, 0, 0, 0, 0, 0, 2 ** (-0.5)],
            ],
            device=device,
        ).detach()

    return change_mat


def irreps_sum(ang_mom: int) -> int:
    """
    Returns the sum of the dimensions of the irreps up to the specified angular momentum.

    :param ang_mom: max angular momenttum to sum up dimensions of irreps
    """
    total = 0
    for i in range(ang_mom + 1):
        total += 2 * i + 1

    return total


def reshape_stress(L0out, L2out, batch_size=1):
    _max_rank = 2
    pred_irreps = torch.zeros(
        (batch_size, irreps_sum(_max_rank)),
        device = L0out.device,
    )
    # L=0
    L=0
    pred_irreps[: ,irreps_sum(L-1): irreps_sum(L)] = L0out.view(batch_size, -1)
    
    L=2
    pred_irreps[: ,irreps_sum(L-1): irreps_sum(L)] = L2out.view(batch_size, -1) 
    
    pred = torch.einsum(
        "ba, cb->ca",
        cg_change_mat(_max_rank, device = L0out.device),
        pred_irreps,
    )
    
    return pred.view(batch_size, 3,3)


class Sphere(nn.Module):
    
    def __init__(self, lmax=2):
        super(Sphere, self).__init__()
        self.lmax = lmax
        
    def forward(self, edge_vec):
        edge_sh = self._spherical_harmonics(self.lmax, edge_vec[..., 0], edge_vec[..., 1], edge_vec[..., 2])
        return edge_sh
        
    @staticmethod
    def _spherical_harmonics(lmax: int, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        sh_0_0 = torch.ones_like(x)
        if lmax == 0:
            return torch.stack([ sh_0_0, ], dim=-1)
        
        sh_1_0, sh_1_1, sh_1_2 = x, y, z
        
        if lmax == 1:
            return torch.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2], dim=-1)

        sh_2_0 = math.sqrt(3.0) * x * z
        sh_2_1 = math.sqrt(3.0) * x * y
        y2 = y.pow(2)
        x2z2 = x.pow(2) + z.pow(2)
        sh_2_2 = y2 - 0.5 * x2z2
        sh_2_3 = math.sqrt(3.0) * y * z
        sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))

        if lmax == 2:
            return torch.stack([sh_0_0, sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4], dim=-1)
