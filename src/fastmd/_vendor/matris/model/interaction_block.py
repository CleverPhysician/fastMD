from __future__ import annotations

import os
import torch
from torch import Tensor, nn
from typing import Any, Dict
from .functions import (
    MLP,
    GatedMLP,
    aggregate,
    get_normalization,
    Dimwise_softmax,
    model_fusions_enabled,
    set_model_fusions_enabled,
)
from torch.utils.checkpoint import checkpoint

THRESHOLD_VALUE = 60000 # Safe value for MatRIS-10M (A100-80GB)

_FUSED_WEIGHTED_SEGMENT_EAGER = os.getenv(
    "MATRIS_FUSED_WEIGHTED_SEGMENT_EAGER", "0"
) == "1"
_FUSED_GATHER_CAT_EAGER = os.getenv("MATRIS_FUSED_GATHER_CAT_EAGER", "0") == "1"
_FUSED_DIRECTED_PAIR_AGGREGATE = os.getenv(
    "MATRIS_FUSED_DIRECTED_PAIR_AGGREGATE", "1"
) != "0"
_FUSED_DIRECTED_PAIR_EXPAND = os.getenv(
    "MATRIS_FUSED_DIRECTED_PAIR_EXPAND", "1"
) != "0"
_FUSED_SEGMENT_ATTENTION_ENABLED = os.getenv(
    "MATRIS_FUSED_SEGMENT_ATTENTION", "1"
) != "0"
_FUSED_SEGMENT_ATTENTION_EAGER = os.getenv(
    "MATRIS_FUSED_SEGMENT_ATTENTION_EAGER", "0"
) == "1"
_FUSED_SEGMENT_ATTENTION_MIN_ROWS = int(
    os.getenv("MATRIS_FUSED_SEGMENT_ATTENTION_MIN_ROWS", "0")
)
_FUSED_PAIRED_SEGMENT_ATTENTION = os.getenv(
    "MATRIS_FUSED_PAIRED_SEGMENT_ATTENTION", "1"
) != "0"
_FUSED_SEGMENT_ATTENTION_REUSE_CSR = os.getenv(
    "MATRIS_FUSED_SEGMENT_ATTENTION_REUSE_CSR", "1"
) != "0"
_FUSED_SEGMENT_ATTENTION_SORTED_TARGET_CSR = os.getenv(
    "MATRIS_FUSED_SEGMENT_ATTENTION_SORTED_TARGET_CSR", "1"
) != "0"
_FUSED_LINE_ENVELOPE_ENABLED = os.getenv("MATRIS_FUSED_LINE_ENVELOPE", "1") != "0"
_FUSED_LINE_ENVELOPE_EAGER = os.getenv(
    "MATRIS_FUSED_LINE_ENVELOPE_EAGER", "0"
) == "1"
_FUSED_RESIDUAL_ADD_ENABLED = os.getenv("MATRIS_FUSED_RESIDUAL_ADD", "1") != "0"
_FUSED_RESIDUAL_ADD_MIN_ROWS = int(
    os.getenv("MATRIS_FUSED_RESIDUAL_ADD_MIN_ROWS", "0")
)

_FUSED_WEIGHTED_SEGMENT_GRAPH = False
_FUSED_GATHER_CAT_GRAPH = False
_FUSED_SEGMENT_ATTENTION_GRAPH = False
_FUSED_LINE_ENVELOPE_GRAPH = False


def set_fused_graph_optimizations(enabled: bool) -> tuple[bool, bool, bool, bool, bool]:
    """Enable graph-safe generic fusions and return the previous state."""
    global _FUSED_WEIGHTED_SEGMENT_GRAPH, _FUSED_GATHER_CAT_GRAPH
    global _FUSED_SEGMENT_ATTENTION_GRAPH, _FUSED_LINE_ENVELOPE_GRAPH
    previous = (
        _FUSED_WEIGHTED_SEGMENT_GRAPH,
        _FUSED_GATHER_CAT_GRAPH,
        _FUSED_SEGMENT_ATTENTION_GRAPH,
        _FUSED_LINE_ENVELOPE_GRAPH,
        set_model_fusions_enabled(enabled),
    )
    _FUSED_WEIGHTED_SEGMENT_GRAPH = bool(enabled)
    _FUSED_GATHER_CAT_GRAPH = bool(enabled)
    _FUSED_SEGMENT_ATTENTION_GRAPH = bool(enabled)
    _FUSED_LINE_ENVELOPE_GRAPH = bool(enabled)
    return previous


def restore_fused_graph_optimizations(
    state: tuple[bool, bool, bool, bool, bool]
) -> None:
    global _FUSED_WEIGHTED_SEGMENT_GRAPH, _FUSED_GATHER_CAT_GRAPH
    global _FUSED_SEGMENT_ATTENTION_GRAPH, _FUSED_LINE_ENVELOPE_GRAPH
    model_fusions_state = state[4]
    (
        _FUSED_WEIGHTED_SEGMENT_GRAPH,
        _FUSED_GATHER_CAT_GRAPH,
        _FUSED_SEGMENT_ATTENTION_GRAPH,
        _FUSED_LINE_ENVELOPE_GRAPH,
    ) = state[:4]
    set_model_fusions_enabled(model_fusions_state)


def _use_fused_weighted_segment_sum(weight: Tensor, value: Tensor) -> bool:
    if not model_fusions_enabled():
        return False
    if not (weight.is_cuda and value.is_cuda and weight.shape == value.shape):
        return False
    return _FUSED_WEIGHTED_SEGMENT_EAGER or _FUSED_WEIGHTED_SEGMENT_GRAPH


def _use_fused_gather_cat(aligned: Tensor, gathered: Tensor) -> bool:
    if not model_fusions_enabled():
        return False
    if not (
        aligned.is_cuda
        and gathered.is_cuda
        and aligned.dim() == 2
        and gathered.dim() == 2
    ):
        return False
    if aligned.shape[1] != gathered.shape[1] or aligned.shape[0] < THRESHOLD_VALUE:
        return False
    return _FUSED_GATHER_CAT_EAGER or _FUSED_GATHER_CAT_GRAPH


def _use_fused_gather_cat4(
    aligned: Tensor, gathered_a: Tensor, gathered_b: Tensor
) -> bool:
    if not _use_fused_gather_cat(aligned, gathered_a):
        return False
    return (
        gathered_b.is_cuda
        and gathered_b.dim() == 2
        and aligned.shape[1] == gathered_b.shape[1]
    )


def _use_fused_segment_attention(logits: Tensor, value: Tensor) -> bool:
    if not model_fusions_enabled() or not _FUSED_SEGMENT_ATTENTION_ENABLED:
        return False
    if not (logits.is_cuda and value.is_cuda and logits.shape == value.shape):
        return False
    if logits.dim() != 2 or logits.shape[0] < _FUSED_SEGMENT_ATTENTION_MIN_ROWS:
        return False
    return _FUSED_SEGMENT_ATTENTION_EAGER or _FUSED_SEGMENT_ATTENTION_GRAPH


def _use_fused_line_envelope(base: Tensor, source: Tensor, target: Tensor) -> bool:
    if not model_fusions_enabled() or not _FUSED_LINE_ENVELOPE_ENABLED:
        return False
    if not (base.is_cuda and source.is_cuda and target.is_cuda):
        return False
    if base.dim() != 2 or source.numel() != target.numel():
        return False
    if source.numel() < THRESHOLD_VALUE:
        return False
    return _FUSED_LINE_ENVELOPE_EAGER or _FUSED_LINE_ENVELOPE_GRAPH


def _segment_attention_csr_metadata(
    graph: Dict,
    key_prefix: str,
    segment: Tensor,
    num_segment: int,
) -> tuple[Tensor | None, Tensor, bool]:
    perm_key = f"_{key_prefix}_segment_attention_perm"
    off_key = f"_{key_prefix}_segment_attention_offsets"
    identity_key = f"_{key_prefix}_segment_attention_identity_perm"
    count_key = f"_{key_prefix}_segment_attention_counts"
    identity_perm = (
        _FUSED_SEGMENT_ATTENTION_SORTED_TARGET_CSR
        and bool(graph.get(identity_key, False))
    )
    if identity_perm:
        off = graph.get(off_key)
        if off is None:
            counts = graph[count_key]
            off = torch.cat((counts.new_zeros(1), counts.cumsum(0))).to(torch.int32)
            graph[off_key] = off
        return None, off, True

    perm = graph.get(perm_key)
    off = graph.get(off_key)
    if perm is None or off is None:
        from .op.triton_segment_attention import segment_attention_csr_metadata

        perm, off = segment_attention_csr_metadata(segment, num_segment)
        graph[perm_key] = perm
        graph[off_key] = off
    return perm, off, False


def _directed_pair_index(graph: Dict, directed2undirected: Tensor) -> Tensor:
    pair_key = "_directed2undirected_pair_index"
    pair_index = graph.get(pair_key)
    if pair_index is None:
        num_undirected = graph["num_undirected"]
        pair_index = torch.argsort(directed2undirected).reshape(num_undirected, 2)
        graph[pair_key] = pair_index.contiguous()
    return graph[pair_key]


def _directed_pair_average_or_aggregate(
    data: Tensor,
    directed2undirected: Tensor,
    graph: Dict,
) -> Tensor:
    num_undirected = graph["num_undirected"]
    if (
        model_fusions_enabled()
        and _FUSED_DIRECTED_PAIR_AGGREGATE
        and data.is_cuda
        and directed2undirected.is_cuda
        and data.dim() == 2
        and directed2undirected.numel() == 2 * num_undirected
    ):
        from .op.triton_directed_pair_aggregate import directed_pair_average

        return directed_pair_average(
            data, _directed_pair_index(graph, directed2undirected)
        )
    return aggregate(
        data=data,
        segment=directed2undirected,
        bin_count=None,
        average=True,
        num_segment=num_undirected,
    )


def _undirected_pair_expand_or_index_select(
    data: Tensor,
    directed2undirected: Tensor,
    graph: Dict,
) -> Tensor:
    num_undirected = graph["num_undirected"]
    if (
        model_fusions_enabled()
        and _FUSED_DIRECTED_PAIR_EXPAND
        and data.is_cuda
        and directed2undirected.is_cuda
        and data.dim() == 2
        and data.shape[0] == num_undirected
        and directed2undirected.numel() == 2 * num_undirected
    ):
        from .op.triton_directed_pair_aggregate import undirected_pair_expand

        return undirected_pair_expand(
            data,
            _directed_pair_index(graph, directed2undirected),
            directed2undirected.numel(),
        )
    return torch.index_select(data, 0, directed2undirected)


def _weighted_segment_sum_or_aggregate(
    weight: Tensor,
    value: Tensor,
    segment: Tensor,
    bin_count: Tensor,
    num_segment: int,
) -> Tensor:
    if _use_fused_weighted_segment_sum(weight, value):
        from .op.triton_weighted_segment_sum import weighted_segment_sum

        return weighted_segment_sum(weight, value, segment, num_segment)
    return aggregate(
        data=weight * value,
        segment=segment,
        bin_count=bin_count,
        average=False,
        num_segment=num_segment,
    )


def _residual_add_or_torch(delta: Tensor, residual: Tensor, weight: Tensor) -> Tensor:
    if (
        model_fusions_enabled()
        and _FUSED_RESIDUAL_ADD_ENABLED
        and delta.is_cuda
        and residual.is_cuda
        and weight.is_cuda
        and delta.dim() == 2
        and residual.shape == delta.shape
        and delta.shape[0] >= _FUSED_RESIDUAL_ADD_MIN_ROWS
        and delta.shape[1] == weight.numel()
        and delta.dtype == residual.dtype == weight.dtype
        and not weight.requires_grad
    ):
        from .op.triton_residual_add import residual_add

        return residual_add(delta, residual, weight)
    return delta + weight * residual

class Graph_Attention_Layer(nn.Module):
    
    def __init__(
        self,
        node_feat_dim: int = 128,
        edge_feat_dim: int = 128,
        hidden_dim: int = 128,
        use_bias: bool = False,
        dropout: float = 0.0,
        mlp_type: str = "GateMLP", # MLP, GateMLP
        activation_type: str = "silu",
        norm_type: str = "layer",
        use_fp16: bool = False, 
    ):
        super().__init__()
        
        self.source_weight_linear = nn.Linear(
            in_features = edge_feat_dim, out_features = edge_feat_dim, bias = False
        )
        self.target_weight_linear = nn.Linear(
            in_features = edge_feat_dim, out_features = edge_feat_dim, bias = False
        )
        if mlp_type.lower() == "mlp":
            self.node_nonlinear_update = nn.Sequential(
                MLP(
                    input_dim=edge_feat_dim * 2 + node_feat_dim,
                    hidden_dim=hidden_dim,
                    output_dim=node_feat_dim,
                    dropout=dropout,
                    bias=use_bias,
                    activation=activation_type,
                ),
                get_normalization(name=norm_type, dim=node_feat_dim) 
            )
            self.edge_nonlinear_update = nn.Sequential(
                MLP(
                    input_dim=node_feat_dim * 2 + edge_feat_dim,
                    hidden_dim=hidden_dim,
                    output_dim=edge_feat_dim,
                    dropout=dropout,
                    bias=use_bias,
                    activation=activation_type,
                    use_fp16=use_fp16,
                ),
                get_normalization(name=norm_type, dim=edge_feat_dim) 
            )
        elif mlp_type.lower() == "gatemlp":
            self.node_nonlinear_update = GatedMLP(
                input_dim=edge_feat_dim * 2 + node_feat_dim,
                hidden_dim=hidden_dim,
                output_dim=node_feat_dim,
                norm_type=norm_type,
                dropout=dropout,
                activation=activation_type,
            )
            self.edge_nonlinear_update = GatedMLP(
                input_dim=node_feat_dim * 2 + edge_feat_dim,
                hidden_dim=hidden_dim,
                output_dim=edge_feat_dim,
                norm_type=norm_type,
                dropout=dropout,
                activation=activation_type,
                use_fp16=use_fp16,
            )
        else:
            raise NotImplementedError

        self.node_res_weight = torch.nn.Parameter(torch.ones(1, node_feat_dim), requires_grad=True)
        self.edge_res_weight = torch.nn.Parameter(torch.ones(1, edge_feat_dim), requires_grad=True)
    
    def forward(self, 
        node_feat: Tensor, 
        edge_feat: Tensor, 
        graph: Dict, # atom graph or line graph
        directed2undirected: Tensor = None,
    ): 
        source_node_index = graph['source_index']
        target_node_index = graph['target_index']
        if directed2undirected is not None:
            source_node_feat = torch.index_select(node_feat, 0, source_node_index)
            target_node_feat = torch.index_select(node_feat, 0, target_node_index)
            # Atom Graph Update
            edge_feat_0 = _undirected_pair_expand_or_index_select(
                edge_feat, directed2undirected, graph
            ) # [edge, dim] -> [2*edge, dim]
            attn_edge_feat = torch.cat(
                [edge_feat_0, target_node_feat, source_node_feat], dim=1
            )
        elif _use_fused_gather_cat(edge_feat, node_feat):
            edge_feat_0 = edge_feat
            from .op.triton_gather_cat import aligned_gather_cat3

            attn_edge_feat = aligned_gather_cat3(
                edge_feat_0, node_feat, target_node_index, source_node_index
            )
        else:
            source_node_feat = torch.index_select(node_feat, 0, source_node_index)
            target_node_feat = torch.index_select(node_feat, 0, target_node_index)
            # Line Graph Update
            edge_feat_0 = edge_feat
            attn_edge_feat = torch.cat(
                [edge_feat_0, target_node_feat, source_node_feat], dim=1
            )
        attn_edge_feat = self.edge_nonlinear_update(attn_edge_feat)

        # ======= update atom feature ======= 
        source_alpha_0 = self.source_weight_linear(edge_feat_0)
        target_alpha_0 = self.target_weight_linear(edge_feat_0)
        
        # Pass num_segment explicitly to avoid compile graph breaks here.
        num_segment = node_feat.shape[0]
        attn_edge_feat_direct = attn_edge_feat
        if directed2undirected is not None:
            attn_edge_feat = _directed_pair_average_or_aggregate(
                attn_edge_feat, directed2undirected, graph
            ) #[2*edge, dim] -> [edge, dim]
        # Compute Attention output
        if _use_fused_segment_attention(source_alpha_0, attn_edge_feat_direct):
            from .op.triton_segment_attention import segment_attention

            if _FUSED_SEGMENT_ATTENTION_REUSE_CSR:
                source_perm, source_off, source_identity = (
                    _segment_attention_csr_metadata(
                        graph, "source", source_node_index, num_segment
                    )
                )
                target_perm, target_off, target_identity = (
                    _segment_attention_csr_metadata(
                        graph, "target", target_node_index, num_segment
                    )
                )
            else:
                source_perm = source_off = target_perm = target_off = None
                source_identity = target_identity = False

            if _FUSED_PAIRED_SEGMENT_ATTENTION:
                from .op.triton_segment_attention import paired_segment_attention

                attn_source_feat, attn_target_feat = paired_segment_attention(
                    source_alpha_0,
                    target_alpha_0,
                    attn_edge_feat_direct,
                    source_node_index,
                    target_node_index,
                    num_segment,
                    source_perm,
                    source_off,
                    target_perm,
                    target_off,
                    source_identity,
                    target_identity,
                )
            else:
                attn_source_feat = segment_attention(
                    source_alpha_0,
                    attn_edge_feat_direct,
                    source_node_index,
                    num_segment,
                    source_perm,
                    source_off,
                    source_identity,
                )
                attn_target_feat = segment_attention(
                    target_alpha_0,
                    attn_edge_feat_direct,
                    target_node_index,
                    num_segment,
                    target_perm,
                    target_off,
                    target_identity,
                )
        else:
            source_alpha = Dimwise_softmax(
                source_alpha_0, source_node_index, num_segment
            )
            target_alpha = Dimwise_softmax(
                target_alpha_0, target_node_index, num_segment
            )
            attn_source_feat = _weighted_segment_sum_or_aggregate(
                source_alpha,
                attn_edge_feat_direct,
                source_node_index,
                graph['source_bincount'],
                len(node_feat),
            )
            attn_target_feat = _weighted_segment_sum_or_aggregate(
                target_alpha,
                attn_edge_feat_direct,
                target_node_index,
                graph['target_bincount'],
                len(node_feat),
            )

        fusion_node_feat = torch.cat([node_feat, attn_target_feat, attn_source_feat], dim=1)
        attn_node_feat = None
        node_residual_fused = False
        if hasattr(self.node_nonlinear_update, "forward_with_residual"):
            attn_node_feat = self.node_nonlinear_update.forward_with_residual(
                fusion_node_feat, node_feat, self.node_res_weight
            )
            node_residual_fused = attn_node_feat is not None
        if attn_node_feat is None:
            attn_node_feat = self.node_nonlinear_update(fusion_node_feat)
        
        # Resdual
        if not node_residual_fused:
            attn_node_feat = _residual_add_or_torch(
                attn_node_feat, node_feat, self.node_res_weight
            )
        attn_edge_feat = _residual_add_or_torch(
            attn_edge_feat, edge_feat, self.edge_res_weight
        )

        return attn_node_feat, attn_edge_feat


class Refinement(nn.Module):
    
    def __init__(
        self,
        node_feat_dim: int = 128,
        edge_feat_dim: int = 128,
        hidden_dim: int = 128,
        num_basis: int = 7,
        dropout: float = 0.0,
        mlp_type: str = "GateMLP",    
        activation_type: str = "silu",
        norm_type: str = "layer",
        use_bias: bool = False,
        graph_type: Literal["atom graph", "line graph"] = "atom graph",
        atom_feat_dim: int = 128,
        use_smoothed_for_delta_edge: bool = False,
        use_fp16: bool = False, 
    ):
        super().__init__()
        self.graph_type = graph_type
        self.use_smoothed_for_delta_edge = use_smoothed_for_delta_edge
        
        if graph_type == "atom graph":
            input_dim = 2 * node_feat_dim + edge_feat_dim
        else:
            input_dim = atom_feat_dim + 2 * node_feat_dim + edge_feat_dim 
        if mlp_type.lower() == "mlp":
            self.edge_nonlinear_update = nn.Sequential(
                MLP(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=edge_feat_dim,
                    dropout=dropout,
                    bias=use_bias,
                    activation=activation_type,
                    use_fp16=use_fp16,
                ),
                get_normalization(name=norm_type, dim=edge_feat_dim)
            )
        elif mlp_type.lower() == "gatemlp":
            self.edge_nonlinear_update = GatedMLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=edge_feat_dim,
                dropout=dropout,
                norm_type=norm_type,
                activation=activation_type,
                use_fp16=use_fp16,
            )
        else:
            raise NotImplementedError
        
        self.node_FFN = MLP(
            input_dim=edge_feat_dim,
            hidden_dim=node_feat_dim,
            output_dim=node_feat_dim,
            bias=use_bias,
        )
        self.edge_FFN = MLP(
            input_dim=edge_feat_dim,
            hidden_dim=edge_feat_dim,
            output_dim=edge_feat_dim,
            bias=use_bias,
            use_fp16=use_fp16,
        )
        self.learnable_envelope = nn.Linear(
            in_features = num_basis, out_features = edge_feat_dim, bias = False
        )
        
        self.node_res_weight = torch.nn.Parameter(torch.ones(1, node_feat_dim), requires_grad=True)
        self.edge_res_weight = torch.nn.Parameter(torch.ones(1, edge_feat_dim), requires_grad=True)

    def forward(
        self,
        node_feat: Tensor,
        edge_feat: Tensor,
        smooth_weight: Tensor,
        graph: Dict,
        directed2undirected: Tensor = None,
        atom_feat: Tensor = None, # Line graph
    ) -> Tensor:
        # Gather
        # when graph=="line graph", make sure atom_deat is not None.
        is_atom_graph = (self.graph_type == "atom graph")
        
        if is_atom_graph: 
            edge_feat_0 = _undirected_pair_expand_or_index_select(
                edge_feat, directed2undirected, graph
            )
        else:
            edge_feat_0 = edge_feat

        source_node_index = graph['source_index']
        target_node_index = graph['target_index']
        # Envelope 
        if is_atom_graph:
            source_node_feat = torch.index_select(node_feat, 0, source_node_index)
            target_node_feat = torch.index_select(node_feat, 0, target_node_index)
            smooth_weight = _undirected_pair_expand_or_index_select(
                smooth_weight, directed2undirected, graph
            )
            smooth_weight = self.learnable_envelope(smooth_weight)
            # Fusion feature
            refine_fusion_feat = torch.cat([edge_feat_0, target_node_feat, source_node_feat], dim=1) 
        else:
            base_envelope = self.learnable_envelope(smooth_weight) 
            if _use_fused_line_envelope(
                base_envelope, source_node_index, target_node_index
            ):
                from .op.triton_line_envelope import line_envelope_product

                smooth_weight = line_envelope_product(
                    base_envelope, source_node_index, target_node_index
                )
            else:
                base_weights_i = torch.index_select(
                    base_envelope, 0, source_node_index
                )
                base_weights_j = torch.index_select(
                    base_envelope, 0, target_node_index
                )
                smooth_weight = base_weights_i * base_weights_j
            # Fusion feature
            if atom_feat is not None and _use_fused_gather_cat4(
                edge_feat_0, atom_feat, node_feat
            ):
                from .op.triton_gather_cat import aligned_gather_cat4

                refine_fusion_feat = aligned_gather_cat4(
                    edge_feat_0,
                    atom_feat,
                    graph['atom_list'],
                    node_feat,
                    target_node_index,
                    source_node_index,
                )
            else:
                source_node_feat = torch.index_select(
                    node_feat, 0, source_node_index
                )
                target_node_feat = torch.index_select(
                    node_feat, 0, target_node_index
                )
                three_body_atom_feat = torch.index_select(
                    atom_feat, 0, graph['atom_list']
                )
                refine_fusion_feat = torch.cat(
                    [
                        edge_feat_0,
                        three_body_atom_feat,
                        target_node_feat,
                        source_node_feat,
                    ],
                    dim=1,
                )
        
        # Nonlinear            
        refine_fusion_feat_nonlinear = self.edge_nonlinear_update(refine_fusion_feat)
        if is_atom_graph and self.use_smoothed_for_delta_edge:
            refine_fusion_feat_smooth = refine_fusion_feat_nonlinear * smooth_weight
            refine_node_feas = aggregate(
                refine_fusion_feat_smooth,
                target_node_index,
                graph['target_bincount'],
                average=False,
                num_segment=len(node_feat),
            )
            input2edgeFFN = refine_fusion_feat_smooth
        else:
            refine_node_feas = _weighted_segment_sum_or_aggregate(
                smooth_weight,
                refine_fusion_feat_nonlinear,
                target_node_index,
                graph['target_bincount'],
                len(node_feat),
            )
            input2edgeFFN = refine_fusion_feat_nonlinear
        
        delta_node_feat = self.node_FFN(refine_node_feas)
        delta_edge_feat = self.edge_FFN(input2edgeFFN)
        
        if is_atom_graph:
            delta_edge_feat = _directed_pair_average_or_aggregate(
                delta_edge_feat, directed2undirected, graph
            ) # [2*edge, dim] -> [edge, dim]

        update_node_feat = _residual_add_or_torch(
            delta_node_feat, node_feat, self.node_res_weight
        )
        update_edge_feat = _residual_add_or_torch(
            delta_edge_feat, edge_feat, self.edge_res_weight
        )
        
        return update_node_feat, update_edge_feat


class Interaction_Block(nn.Module):
    """
    Interaction Block for MatRIS that processes both atom graphs and line graphs.
    
    This block performs attention-based message passing and refinement on two hierarchical graph structures:
    1. Atom graph: Nodes represent atoms, edges represent bonds
    2. Line graph: Nodes represent bonds, edges represent three-body interactions (angles)
    
    Attributes:
        attn_block_atom_graph (Graph_Attention_Layer): Attention layer for atom graph
        attn_block_line_graph (Graph_Attention_Layer): Attention layer for line graph
        refine_block_atom_graph (Refinement): Refinement layer for atom graph  
        refine_block_line_graph (Refinement): Refinement layer for line graph
    """
    
    def __init__(self,
                 node_feat_dim: int = 128,
                 edge_feat_dim: int = 128,
                 three_body_feat_dim: int = 128,
                 num_radial: int = 7,
                 num_angular: int = 7,
                 dropout: float = 0.0, 
                 use_bias: bool = False,
                 use_smoothed_for_delta_edge: bool = False,
                 mlp_type: str = "GateMLP",
                 norm_type: str = "layer",
                 activation_type: str = "silu",
                 enable_compile: bool = False,
                 enable_checkpoint: bool | None = None,
                 ):
        """
        Initialize the Interaction Block.

        Args:
            node_feat_dim (int): Dimension of node features (atom features)
            edge_feat_dim (int): Dimension of edge features (bond features)  
            three_body_feat_dim (int): Dimension of three-body features (angle features)
            mlp_type (str): Type of MLP to use in the layers
            norm_type (str): Type of normalization to apply
            activation_type (str): Type of activation function to use
            enable_compile (bool): Whether to compile the inner attention and
                refinement blocks.
            enable_checkpoint (bool | None): True forces checkpointing on,
                False forces it off, and None keeps the automatic threshold.
        """
        super().__init__()
        self.enable_compile = enable_compile
        self.enable_checkpoint = enable_checkpoint
        
        self.attn_block_atom_graph = Graph_Attention_Layer(
                node_feat_dim=node_feat_dim,
                edge_feat_dim=edge_feat_dim,
                hidden_dim=node_feat_dim,
                use_bias=use_bias,
                mlp_type=mlp_type,
                norm_type=norm_type,
                activation_type=activation_type,
            )

        self.attn_block_line_graph = Graph_Attention_Layer(
                node_feat_dim=edge_feat_dim,
                edge_feat_dim=three_body_feat_dim,
                hidden_dim=edge_feat_dim,
                use_bias=use_bias,
                mlp_type=mlp_type,
                norm_type=norm_type,
                activation_type=activation_type,
                use_fp16=False,
            )
        
        self.refine_block_atom_graph = Refinement(
                node_feat_dim=node_feat_dim,
                edge_feat_dim=edge_feat_dim,
                hidden_dim=node_feat_dim,  
                num_basis=num_radial,      
                dropout=dropout,            
                activation_type=activation_type,
                norm_type=norm_type,
                use_bias=use_bias,
                mlp_type=mlp_type,
                graph_type="atom graph",
                use_smoothed_for_delta_edge=use_smoothed_for_delta_edge,
            )
        
        self.refine_block_line_graph = Refinement(
                node_feat_dim=edge_feat_dim,
                edge_feat_dim=three_body_feat_dim,
                hidden_dim=edge_feat_dim,  
                num_basis=num_angular,     
                dropout=dropout,          
                activation_type=activation_type,
                norm_type=norm_type,
                use_bias=use_bias,
                mlp_type=mlp_type,
                graph_type="line graph",
                atom_feat_dim=node_feat_dim,
                use_fp16=False, 
            )
    
    def forward(
        self,
        batch_graph: Dict,
        node_feat: Tensor, 
        edge_feat: Tensor, 
        threebody_feat: Tensor | None,
        smooth_weight: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass of the Interaction Block.
        
        Args:
            batch_graph: Graph object containing:
                - atom_graph_dict: Atom graph structure
                - line_graph_dict: Bond graph (line graph) structure  
                - directed2undirected: Mapping from directed to undirected edges
                - bond_bases_bg: Smooth weights for bond graph
                - bond_bases_ag: Smooth weights for atom graph
            node_feat (Tensor): Node features [num_atoms, node_feat_dim]
            edge_feat (Tensor): Edge features [num_bonds, edge_feat_dim] 
            threebody_feat (Tensor): Three-body features [num_angles, three_body_feat_dim] or None
            bincount_atom_graph (Dict): Bincount information for atom graph
            bincount_line_graph (Dict): Bincount information for line graph
        """
        # Initialize variables to handle both cases (with and without threebody features)
        attn_edge_feat = edge_feat 
        attn_threebody_feat = threebody_feat
        update_edge_feat = edge_feat
        update_threebody_feat = threebody_feat 
        enable_checkpoint = self.enable_checkpoint
        if enable_checkpoint is None:
            use_checkpoint = (
                isinstance(threebody_feat, torch.Tensor)
                and threebody_feat.shape[0] > THRESHOLD_VALUE
            )
        else:
            use_checkpoint = enable_checkpoint
        if self.enable_compile and not getattr(self, "_is_compiled", False):
            self.attn_block_atom_graph = torch.compile(
                self.attn_block_atom_graph, dynamic=True
            )
            self.attn_block_line_graph = torch.compile(
                self.attn_block_line_graph, dynamic=True
            )
            self.refine_block_atom_graph = torch.compile(
                self.refine_block_atom_graph, dynamic=True
            )
            self.refine_block_line_graph = torch.compile(
                self.refine_block_line_graph, dynamic=True
            )
            self._is_compiled = True

        # Process line graph (bond graph) with attention if threebody features exist
        if threebody_feat is not None: 
            attn_edge_feat, attn_threebody_feat = self.wrapper_attn_layer(
                attn_layer=self.attn_block_line_graph,
                node_feat=edge_feat,
                edge_feat=threebody_feat,
                graph=batch_graph['line_graph_dict'],
                use_checkpoint=use_checkpoint, 
            )

        # Process atom graph with attention
        attn_node_feat, attn_edge_feat = self.wrapper_attn_layer(
            attn_layer=self.attn_block_atom_graph,
            node_feat=node_feat, 
            edge_feat=attn_edge_feat, 
            graph=batch_graph['atom_graph_dict'], 
            directed2undirected=batch_graph['directed2undirected'],
        ) 
        
        # Refine line graph features if threebody features exist
        if threebody_feat is not None:
            update_edge_feat, update_threebody_feat = self.wrapper_refine_layer(
                refine_layer=self.refine_block_line_graph,
                node_feat=attn_edge_feat,
                edge_feat=attn_threebody_feat,
                smooth_weight=smooth_weight['line graph'],
                graph=batch_graph['line_graph_dict'],
                atom_feat=attn_node_feat,
                use_checkpoint=use_checkpoint,
            )
        
        # Refine atom graph features
        update_node_feat, update_edge_feat = self.wrapper_refine_layer(
            refine_layer=self.refine_block_atom_graph,
            node_feat=attn_node_feat,
            edge_feat=update_edge_feat,
            smooth_weight=smooth_weight['atom graph'],
            graph=batch_graph['atom_graph_dict'],
            directed2undirected=batch_graph['directed2undirected'],
        )
        
        return update_node_feat, update_edge_feat, update_threebody_feat
    
    def wrapper_attn_layer(self,
                            attn_layer: nn.Module,
                            node_feat: Tensor, 
                            edge_feat: Tensor, 
                            graph: Dict,
                            directed2undirected: Tensor = None,
                            use_checkpoint: bool = False,
                       ):
        if use_checkpoint:
            attn_node_feat, attn_edge_feat = checkpoint(
                attn_layer,
                node_feat, 
                edge_feat, 
                graph,
                directed2undirected,
                use_reentrant=False,
            ) 
        else:
            attn_node_feat, attn_edge_feat = attn_layer(
                node_feat=node_feat, 
                edge_feat=edge_feat, 
                graph=graph,
                directed2undirected=directed2undirected,
            )
        
        return attn_node_feat, attn_edge_feat 
        
    def wrapper_refine_layer(self, 
                            refine_layer: nn.Module,
                            node_feat: Tensor,
                            edge_feat: Tensor,
                            smooth_weight: Tensor,
                            graph: Dict,
                            directed2undirected: Tensor = None,
                            atom_feat: Tensor = None,
                            use_checkpoint: bool = False,
                        ):
        if use_checkpoint:
            update_node_feat, update_edge_feat = checkpoint(
                refine_layer,
                node_feat,
                edge_feat,
                smooth_weight,
                graph,
                directed2undirected,
                atom_feat,
                use_reentrant=False,
            )
        else:
            update_node_feat, update_edge_feat = refine_layer(
                    node_feat=node_feat,
                    edge_feat=edge_feat,
                    smooth_weight=smooth_weight,
                    graph=graph,
                    directed2undirected=directed2undirected,
                    atom_feat=atom_feat,
                )
        return update_node_feat, update_edge_feat 
        
    
