from __future__ import annotations

import os

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .basis_function import (
    FourierExpansion, 
    BesselExpansion, 
    GaussianExpansion,
    GaussianExpansion,
    SphericalExpansion,
    SinusoidalTimeExpansion
)
from .functions import get_normalization, SwishLayer, aggregate, model_fusions_enabled

_FUSED_THREEBODY_LINEAR_ENABLED = os.getenv(
    "MATRIS_FUSED_THREEBODY_LINEAR", "1"
) != "0"
_FUSED_THREEBODY_LINEAR_EAGER = os.getenv(
    "MATRIS_FUSED_THREEBODY_LINEAR_EAGER", "0"
) == "1"
_FUSED_THREEBODY_LINEAR_GRAPH = False
_FUSED_THREEBODY_LINEAR_MIN_ROWS = int(
    os.getenv("MATRIS_FUSED_THREEBODY_LINEAR_MIN_ROWS", "60000")
)
_FUSED_THREEBODY_BASIS_ENABLED = os.getenv(
    "MATRIS_FUSED_THREEBODY_BASIS", "1"
) != "0"
_FUSED_THREEBODY_BASIS_EAGER = os.getenv(
    "MATRIS_FUSED_THREEBODY_BASIS_EAGER", "0"
) == "1"
_FUSED_THREEBODY_BASIS_GRAPH = False
_FUSED_THREEBODY_BASIS_MIN_ROWS = int(
    os.getenv("MATRIS_FUSED_THREEBODY_BASIS_MIN_ROWS", "60000")
)
_UNDIRECTED_EDGE_INIT = os.getenv("MATRIS_UNDIRECTED_EDGE_INIT", "1") != "0"


def set_fused_feature_graph_optimizations(enabled: bool) -> tuple[bool, bool]:
    global _FUSED_THREEBODY_LINEAR_GRAPH, _FUSED_THREEBODY_BASIS_GRAPH
    previous = (_FUSED_THREEBODY_LINEAR_GRAPH, _FUSED_THREEBODY_BASIS_GRAPH)
    _FUSED_THREEBODY_LINEAR_GRAPH = bool(enabled)
    _FUSED_THREEBODY_BASIS_GRAPH = bool(enabled)
    return previous


def restore_fused_feature_graph_optimizations(state: tuple[bool, bool]) -> None:
    global _FUSED_THREEBODY_LINEAR_GRAPH, _FUSED_THREEBODY_BASIS_GRAPH
    (_FUSED_THREEBODY_LINEAR_GRAPH, _FUSED_THREEBODY_BASIS_GRAPH) = state


def _use_fused_threebody_linear(basis: Tensor) -> bool:
    if not model_fusions_enabled() or not _FUSED_THREEBODY_LINEAR_ENABLED:
        return False
    if not (basis.is_cuda and basis.dim() == 2):
        return False
    if basis.shape[0] < _FUSED_THREEBODY_LINEAR_MIN_ROWS:
        return False
    return _FUSED_THREEBODY_LINEAR_EAGER or _FUSED_THREEBODY_LINEAR_GRAPH


def _use_fused_threebody_basis(
    unit_vec: Tensor,
    target_index: Tensor,
    source_index: Tensor,
    freqs: Tensor,
) -> bool:
    if not model_fusions_enabled() or not _FUSED_THREEBODY_BASIS_ENABLED:
        return False
    if not (
        unit_vec.is_cuda
        and target_index.is_cuda
        and source_index.is_cuda
        and freqs.is_cuda
    ):
        return False
    if unit_vec.dim() != 2 or unit_vec.shape[1] != 3:
        return False
    if (
        target_index.numel() != source_index.numel()
        or target_index.numel() < _FUSED_THREEBODY_BASIS_MIN_ROWS
    ):
        return False
    if freqs.requires_grad or freqs.numel() != 3:
        return False
    return _FUSED_THREEBODY_BASIS_EAGER or _FUSED_THREEBODY_BASIS_GRAPH

class AtomTypeEmbedding(nn.Module):
    """Encode an atom by its atomic number using 'nn.Embedding'."""
    
    def __init__(self, atom_feat_dim: int, max_num_elements: int = 94):
        """
        Args:
            atom_feature_dim (int): dimension of atomic embedding.
            max_num_elements (int): maximum number of elements in the dataset.
                Default = 94
        """
        super().__init__()
        self.embedding = nn.Embedding(max_num_elements, atom_feat_dim)
        self.atom_swish_layer = SwishLayer(input_dim = atom_feat_dim,
                                           output_dim = atom_feat_dim,
                                           bias=False)

    def forward(self, atom_type: Tensor) -> Tensor:
        """
        Args:
            atom_type (Tensor): [nAtoms, 1].
        """
        node_feat = self.embedding(atom_type)
        node_feat = self.atom_swish_layer(node_feat)
        return node_feat


class EdgeBasisEmbedding(nn.Module):
    """Embed edge length using 'Bessel' or 'Gaussian'."""
    
    def __init__(
        self,
        pairwise_cutoff: float = 6,
        three_body_cutoff: float = 4,
        num_radial: int = 7,
        edge_feat_dim: int = 128,
        envelope_exponent: int = 8,
        distance_expansion: str = "Bessel",
        learnable: bool = False,
    ):
        """
        Args:
            pairwise_cutoff (float): The cutoff for pairwise interaction.
                Default = 6
            three_body_cutoff (float): The cutoff for three-body interaction. 
                Default = 4
            num_radial (int): The number of radial. 
                Default = 7
            envelope_exponent (int): envelope exponent of Envelope function.
                Default = 8
            distance_expansion (str): The function of basis, "Bessel" or "Gaussian".
                Default = "Bessel"
            learnable(bool): Whether the frequency in bessel expansion is learnable.
                Default = False
        """
        super().__init__()

        self.distance_expansion = distance_expansion
        if self.distance_expansion.lower() == "bessel":
            self.pairwise_rbf_expansion = BesselExpansion(
                num_radial=num_radial,
                cutoff=pairwise_cutoff,
                envelope_exponent=envelope_exponent,
                learnable=learnable,
            )
            self.threebody_rbf_expansion = BesselExpansion(
                num_radial=num_radial,
                cutoff=three_body_cutoff,
                envelope_exponent=envelope_exponent,
                learnable=learnable,
            )
        elif self.distance_expansion.lower() == "gaussian":
            self.pairwise_rbf_expansion = GaussianExpansion(
                start=0.0,
                stop=pairwise_cutoff,
                num_gaussians=num_radial,
                basis_width_scalar=2.0,
            )
            self.threebody_rbf_expansion = GaussianExpansion(
                start=0.0,
                stop=three_body_cutoff,
                num_gaussians=num_radial,
                basis_width_scalar=2.0,
            )
        else:
            raise NotImplementedError

        self.edge_linear1 = nn.Linear(
            in_features=num_radial, out_features=edge_feat_dim, bias=False
        )
        self.edge_linear2 = nn.Linear(
            in_features=edge_feat_dim, out_features=edge_feat_dim, bias=False
        )
        self.swish_layer = SwishLayer(input_dim = edge_feat_dim,
                                           output_dim = edge_feat_dim,
                                           bias=False)
        self.edge_init_norm = get_normalization(name="layer", dim=edge_feat_dim)

        
    def forward(
        self,
        graphs, 
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            center_pos (Tensor): coord of center atoms.
            neighbor_pos (Tensor): coordinates of neighbor atoms.
            undirected2directed (Tensor): mapping from undirected vectors to one of its
                directed vectors. Following repo: https://github.com/CederGroupHub/chgnet
            image (Tensor): the periodic image specifying the location of neighbor atoms.
            lattice (Tensor): the lattice matrix of structure.
        """
         
        # From directed edge to undirected edge, this motivation is inspired by https://github.com/CederGroupHub/chgnet
        unique_edge_lengths = torch.index_select( graphs['edge_lengths'], 0, graphs['undirected2directed'] ) #[nEdges/2]

        pairwise_rbf = self.pairwise_rbf_expansion(unique_edge_lengths)  #[nEdges/2, num_radial]
        threebody_rbf = self.threebody_rbf_expansion(unique_edge_lengths)  #[nEdges/2, num_radial]
        
        edge_feat_undirect = self.edge_linear1(pairwise_rbf)
        if model_fusions_enabled() and _UNDIRECTED_EDGE_INIT:
            # Both directed rows of an undirected pair are identical here, so
            # linear/norm may run once before the redundant expand+average.
            edge_feat = self.edge_linear2(edge_feat_undirect)
            edge_feat = self.edge_init_norm(edge_feat)
        else:
            edge_feat_direct = torch.index_select(
                edge_feat_undirect, 0, graphs['directed2undirected']
            )
            edge_feat_direct = self.edge_linear2(edge_feat_direct)
            edge_feat_direct = self.edge_init_norm(edge_feat_direct)
            edge_feat = aggregate(
                data=edge_feat_direct,
                segment=graphs['directed2undirected'],
                bin_count=None,
                average=True,
                num_segment=graphs['undirected2directed'].shape[0],
            )
        edge_feat = self.swish_layer(edge_feat)
        smooth_weight={"atom graph": pairwise_rbf, "line graph": threebody_rbf}
        #return edge_feat, pairwise_rbf, threebody_rbf
        return edge_feat, smooth_weight


class ThreebodyEmbedding(nn.Module):
    """Embed threebody using 'Fourier' or 'Sphere Harmonic'."""

    def __init__(self, 
                 num_angular: int = 7, # Fourier
                 max_n: int=4, max_l: int=4, cutoff: int=6,  #SH
                 three_body_feat_dim: int = 128,
                 three_body_expansion: str = "fourier",
                 learnable: bool = True):
        super().__init__()
        
        self.three_body_expansion = three_body_expansion.lower()
        if self.three_body_expansion == "fourier":
            max_f = (num_angular - 1) // 2
            self.expansion = FourierExpansion(
                max_f=max_f, learnable=learnable
            )
            self.angle_embedding = nn.Linear(
                in_features=num_angular, out_features=three_body_feat_dim, bias=False
            )
        elif self.three_body_expansion == "sh":
            self.expansion = SphericalExpansion(max_n=max_n, max_l=max_l, cutoff=cutoff)
            self.angle_embedding = nn.Linear(
                in_features=max_n * max_l, out_features=three_body_feat_dim, bias=False
            ) 
        else:
            raise NotImplementedError

        self.swish_layer = SwishLayer(input_dim = three_body_feat_dim,
                                    output_dim = three_body_feat_dim,
                                    bias=False)
        self._combined_swish_cache_key = None
        self._combined_swish_weight = None

    def _combined_swish_linear_weight(self, basis: Tensor) -> Tensor | None:
        if self.three_body_expansion != "fourier" or not _use_fused_threebody_linear(basis):
            return None
        if not isinstance(self.angle_embedding, nn.Linear):
            return None
        swish_linear = self.swish_layer.linear
        if self.angle_embedding.bias is not None or swish_linear.bias is not None:
            return None
        angle_weight = self.angle_embedding.weight
        swish_weight = swish_linear.weight
        if angle_weight.requires_grad or swish_weight.requires_grad:
            return None
        if angle_weight.device != basis.device or swish_weight.device != basis.device:
            return None
        if angle_weight.dtype != basis.dtype or swish_weight.dtype != basis.dtype:
            return None
        key = (
            angle_weight.data_ptr(),
            swish_weight.data_ptr(),
            angle_weight._version,
            swish_weight._version,
            basis.device,
            basis.dtype,
        )
        if self._combined_swish_cache_key != key:
            if basis.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
                return None
            with torch.no_grad():
                self._combined_swish_weight = torch.matmul(
                    swish_weight.detach(), angle_weight.detach()
                ).contiguous()
                self._combined_swish_cache_key = key
        return self._combined_swish_weight
     
    def forward(self, graphs):
        target_de_index = graphs['line_graph_dict']['target_DE_index']
        source_de_index = graphs['line_graph_dict']['source_DE_index']
        unit_edge_vectors = graphs['unit_edge_vectors']
        if (
            self.three_body_expansion == "fourier"
            and _use_fused_threebody_basis(
                unit_edge_vectors,
                target_de_index,
                source_de_index,
                self.expansion.frequencies,
            )
        ):
            from .op.triton_threebody_fourier import threebody_fourier_basis

            three_body_basis = threebody_fourier_basis(
                unit_edge_vectors,
                target_de_index,
                source_de_index,
                self.expansion.frequencies,
                self.expansion.scale,
            )
        else:
            edge_vecs_ij = torch.index_select(
                unit_edge_vectors, 0, target_de_index
            )
            edge_vecs_jk = torch.index_select(
                unit_edge_vectors, 0, source_de_index
            )
            theta_ijk = torch.sum(edge_vecs_ij * edge_vecs_jk, dim=1) * (1 - 1e-6)
            angle = torch.acos(theta_ijk)

            if self.three_body_expansion == "sh":
                edge_vecs_jk_len = torch.norm(edge_vecs_jk, dim=1)
                three_body_basis = self.expansion(edge_vecs_jk_len, angle)
            else:
                three_body_basis = self.expansion(angle)
        
        combined_weight = self._combined_swish_linear_weight(three_body_basis)
        if combined_weight is None:
            threebody_feat = self.angle_embedding(three_body_basis)
            threebody_feat = self.swish_layer(threebody_feat)
        else:
            threebody_feat = F.linear(three_body_basis, combined_weight)
            threebody_feat = self.swish_layer.act(threebody_feat)
        
        return threebody_feat
    

class ThreebodyFourierExpansion(nn.Module):
    """Encode an three-body terms using Fourier Expansion."""
    
    def __init__(self, num_angular: int = 7, learnable: bool = True):
        super().__init__()
        
        assert num_angular % 2 == 1, f"{num_angular=} must be an odd integer"
        max_f = (num_angular - 1) // 2
        self.fourier_expansion = FourierExpansion(
            max_f=max_f, learnable=learnable
        )

    def forward(self, edge_vec_ij: Tensor, edge_vec_jk: Tensor) -> Tensor:
        """
        Args:
            edge_vec_i (Tensor): normalized edge vector ij [n_angle, 3]
            edge_vec_j (Tensor): normalized edge vector jk [n_angle, 3]
        """
        theta_ijk = torch.sum(edge_vec_ij * edge_vec_jk, dim=1) * (1 - 1e-6)
        angle = torch.acos(theta_ijk) 
        return self.fourier_expansion(angle)


class three_bodySHExpansion(nn.Module):
    """Encode an three-body terms using Sphere Harmonic Expansion."""

    def __init__(self, max_n: int=4, max_l: int=4, cutoff: int=6) -> None:
        super().__init__()
        self.sbf = SphericalExpansion(max_n=max_n, max_l=max_l, cutoff=cutoff)
    
    def forward(self, edge_vec_ij: Tensor, edge_vec_jk: Tensor) -> Tensor:
        """
        Args:
            edge_vec_i (Tensor): normalized edge vector ij [n_angle, 3]
            edge_vec_j (Tensor): normalized edge vector jk [n_angle, 3]
        """
        theta_ijk = torch.sum(edge_vec_ij * edge_vec_jk, dim=1) * (1 - 1e-6)
        angle = torch.acos(theta_ijk)
        
        edge_vec_jk_len = torch.norm(edge_vec_jk, dim=1)
        angle_basis = self.sbf(edge_vec_jk_len, angle)
        return angle_basis


class TimeEmbedding(nn.Module):
    """Encode time using nonlinear"""

    def __init__(self, time_feat_dim):
        super().__init__()
        self.time_encode = SinusoidalTimeExpansion(dim = time_feat_dim)
        self.linear1 = nn.Linear(
            in_features = time_feat_dim, out_features = 2 * time_feat_dim
        )
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(
            in_features = time_feat_dim * 2, out_features = time_feat_dim
        )
     
    def forward(self, time):
        time_enc= self.time_encode(time)
        time_enc = self.act(self.linear1(time_enc))
        time_emb = self.linear2(time_enc)
        return time_emb
