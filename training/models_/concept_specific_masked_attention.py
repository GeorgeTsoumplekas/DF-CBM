"""
Concept-Specific Masked Attention Concept Bottleneck Layer.

Expected input shapes:
- patch_tokens: [B, N, D_in]
- region_masks: [B, R, N] or [B, R, H, W]
- concept_region_matrix: [K, R]

Typical CLIP ViT-L/14 setting:
- N = 256 for a 16 x 16 patch grid.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ConceptSpecificMaskedAttentionCBL(nn.Module):
    """Concept-specific masked attention concept bottleneck layer.

    Parameters
    ----------
    input_dim:
        Dimensionality of each CLIP patch token.
    num_concepts:
        Number of concepts K.
    num_regions:
        Number of facial + boundary regions R.
    concept_region_matrix:
        Binary matrix A with shape [K, R], where A[k, r] = 1 means
        concept k is allowed to use region r.
    query_init:
        Optional initial concept query matrix with shape [K, attention_dim].
        This is where CLIP text embeddings of concept names can be passed,
        provided they have the same dimensionality as attention_dim.
    attention_dim:
        Dimension used for keys and concept queries. If None, defaults to input_dim.
    value_dim:
        Dimension used for values. If None, defaults to attention_dim.
    hidden_dim:
        Dimension of the shared concept evidence projection W_s.
        If None, defaults to value_dim.
    shared_proj_bias:
        If True, add a shared bias term to the shared projection layer.
        If False, use weights only (paper default).
    prior_strength:
        Multiplicative strength gamma for the log spatial prior term.
        gamma = 1.0 corresponds directly to log(p_{k,i} + eps).
        gamma = 0.0 disables the spatial prior while keeping the same architecture.
    eps:
        Numerical stability constant used inside log(p + eps).
    normalize_queries:
        If True, L2-normalize concept queries and keys before dot-product attention.
        This can be useful when q_k is initialized from CLIP text embeddings.
    dropout:
        Dropout applied to the hidden concept evidence h_k before the final
        concept-specific classifier.
    freeze_concept_queries:
        If True, concept queries are not updated during training. Useful for
        ablations where queries are initialized from CLIP text embeddings and
        should remain fixed.
    """

    def __init__(
        self,
        input_dim: int,
        num_concepts: int,
        num_regions: int,
        concept_region_matrix: Union[Tensor, list],
        query_init: Optional[Tensor] = None,
        attention_dim: Optional[int] = None,
        value_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        shared_proj_bias: bool = False,
        prior_strength: float = 1.0,
        eps: float = 1e-6,
        normalize_queries: bool = False,
        dropout: float = 0.0,
        classifier_bias_init: Optional[Tensor] = None,
        freeze_concept_queries: bool = False,
    ) -> None:
        super().__init__()

        self.input_dim = int(input_dim)
        self.num_concepts = int(num_concepts)
        self.num_regions = int(num_regions)
        self.attention_dim = int(attention_dim or input_dim)
        self.value_dim = int(value_dim or self.attention_dim)
        self.hidden_dim = int(hidden_dim or self.value_dim)
        self.shared_proj_bias = bool(shared_proj_bias)
        self.prior_strength = float(prior_strength)
        self.eps = float(eps)
        self.normalize_queries = bool(normalize_queries)
        self.freeze_concept_queries = bool(freeze_concept_queries)

        concept_region_matrix = torch.as_tensor(
            concept_region_matrix, dtype=torch.float32
        )
        if concept_region_matrix.shape != (self.num_concepts, self.num_regions):
            raise ValueError(
                "concept_region_matrix must have shape "
                f"[num_concepts, num_regions] = [{self.num_concepts}, {self.num_regions}], "
                f"but got {tuple(concept_region_matrix.shape)}."
            )

        self.register_buffer("concept_region_matrix", concept_region_matrix)

        # Project CLIP patch tokens to key and value spaces.
        self.key_proj = nn.Linear(self.input_dim, self.attention_dim, bias=False)
        self.value_proj = nn.Linear(self.input_dim, self.value_dim, bias=False)

        # Concept queries q_k.
        if query_init is not None:
            query_init = torch.as_tensor(query_init, dtype=torch.float32)
            expected_shape = (self.num_concepts, self.attention_dim)
            if query_init.shape != expected_shape:
                raise ValueError(
                    "query_init must have shape "
                    f"[num_concepts, attention_dim] = {expected_shape}, "
                    f"but got {tuple(query_init.shape)}. "
                    "If CLIP text embeddings have a different dimension, project them "
                    "to attention_dim before passing them to this class."
                )
            self.concept_queries = nn.Parameter(query_init.clone())
            self.register_buffer("initial_concept_queries", query_init.clone())
        else:
            self.concept_queries = nn.Parameter(
                torch.empty(self.num_concepts, self.attention_dim)
            )
            nn.init.normal_(self.concept_queries, mean=0.0, std=0.02)
            self.register_buffer(
                "initial_concept_queries", self.concept_queries.detach().clone()
            )

        if self.freeze_concept_queries:
            self.concept_queries.requires_grad_(False)

        # Shared single-layer MLP: h_k = ReLU(W_s u_k + b_s) when bias is enabled.
        self.shared_proj = nn.Linear(
            self.value_dim, self.hidden_dim, bias=self.shared_proj_bias
        )
        self.activation = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(dropout)

        # Concept-specific linear classifier: w_k^T h_k + b_k.
        self.concept_classifier_weight = nn.Parameter(
            torch.empty(self.num_concepts, self.hidden_dim)
        )
        nn.init.xavier_uniform_(self.concept_classifier_weight)
        if classifier_bias_init is not None:
            bias_init = torch.as_tensor(classifier_bias_init, dtype=torch.float32)
            if bias_init.shape != (self.num_concepts,):
                raise ValueError(
                    "classifier_bias_init must have shape "
                    f"[num_concepts] = [{self.num_concepts}], "
                    f"but got {tuple(bias_init.shape)}."
                )
            self.concept_classifier_bias = nn.Parameter(bias_init.clone())
        else:
            self.concept_classifier_bias = nn.Parameter(torch.zeros(self.num_concepts))

    def forward(
        self,
        patch_tokens: Tensor,
        region_masks: Tensor,
        return_attention: bool = False,
        return_prior: bool = False,
        return_evidence: bool = False,
        return_probs: bool = False,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """Forward pass.

        Parameters
        ----------
        patch_tokens:
            CLIP patch tokens with shape [B, N, D_in].
        region_masks:
            Downsampled facial/boundary masks with shape [B, R, N]
            or [B, R, H, W]. Values should be in [0, 1].
        return_attention:
            If True, include attention maps alpha with shape [B, K, N].
        return_prior:
            If True, include concept-specific spatial priors p with shape [B, K, N].
        return_evidence:
            If True, include concept evidence vectors u_k and hidden features h_k.
        return_probs:
            If True, include sigmoid probabilities. The logits are always returned.

        Returns
        -------
        If all return_* flags are False:
            concept_logits: Tensor of shape [B, K]

        Otherwise:
            Dictionary containing at least:
                "concept_logits": [B, K]
            and optionally:
                "concept_probs": [B, K]
                "attention": [B, K, N]
                "spatial_prior": [B, K, N]
                "concept_evidence": [B, K, value_dim]
                "concept_hidden": [B, K, hidden_dim]
        """
        if patch_tokens.ndim != 3:
            raise ValueError(
                f"patch_tokens must have shape [B, N, D], got {tuple(patch_tokens.shape)}."
            )

        batch_size, num_patches, input_dim = patch_tokens.shape
        if input_dim != self.input_dim:
            raise ValueError(
                f"Expected patch token dim {self.input_dim}, got {input_dim}."
            )

        region_masks = self._prepare_region_masks(region_masks, num_patches)
        if region_masks.shape[0] != batch_size:
            raise ValueError(
                "Batch size mismatch between patch_tokens and region_masks: "
                f"{batch_size} vs {region_masks.shape[0]}."
            )

        # Build concept-specific spatial prior p_{n,k,i}.
        spatial_prior = self.compute_spatial_prior(region_masks)

        # Linear projections.
        keys = self.key_proj(patch_tokens)  # [B, N, d]
        values = self.value_proj(patch_tokens)  # [B, N, value_dim]
        queries = self.concept_queries  # [K, d]

        if self.normalize_queries:
            keys = F.normalize(keys, p=2, dim=-1)
            queries = F.normalize(queries, p=2, dim=-1)

        # alpha_{k,i} = softmax_i( q_k^T W f_i / sqrt(d) + log(p_{k,i} + eps) )
        attention_logits = torch.einsum("kd,bnd->bkn", queries, keys)
        attention_logits = attention_logits / math.sqrt(self.attention_dim)

        if self.prior_strength != 0.0:
            attention_logits = attention_logits + self.prior_strength * torch.log(
                spatial_prior + self.eps
            )

        attention = torch.softmax(attention_logits, dim=-1)  # [B, K, N]

        # Concept-specific evidence u_k.
        concept_evidence = torch.einsum("bkn,bnv->bkv", attention, values)

        # Shared projection and concept-specific classifier.
        concept_hidden = self.activation(self.shared_proj(concept_evidence))
        concept_hidden = self.dropout(concept_hidden)

        concept_logits = (
            concept_hidden * self.concept_classifier_weight.unsqueeze(0)
        ).sum(dim=-1) + self.concept_classifier_bias.unsqueeze(0)

        if not (return_attention or return_prior or return_evidence or return_probs):
            return concept_logits

        output: Dict[str, Tensor] = {"concept_logits": concept_logits}

        if return_probs:
            output["concept_probs"] = torch.sigmoid(concept_logits)

        if return_attention:
            output["attention"] = attention

        if return_prior:
            output["spatial_prior"] = spatial_prior

        if return_evidence:
            output["concept_evidence"] = concept_evidence
            output["concept_hidden"] = concept_hidden

        return output

    def compute_spatial_prior(self, region_masks: Tensor) -> Tensor:
        """Compute p_{n,k,i} = max_{r: A[k,r]=1} m_{n,r,i}.

        Parameters
        ----------
        region_masks:
            Tensor with shape [B, R, N].

        Returns
        -------
        spatial_prior:
            Tensor with shape [B, K, N].
        """
        if region_masks.ndim != 3:
            raise ValueError(
                f"region_masks must have shape [B, R, N], got {tuple(region_masks.shape)}."
            )

        _, num_regions, _ = region_masks.shape
        if num_regions != self.num_regions:
            raise ValueError(f"Expected {self.num_regions} regions, got {num_regions}.")

        # [B, 1, R, N]
        masks = region_masks.unsqueeze(1)

        # [1, K, R, 1]
        allowed = (
            self.concept_region_matrix.to(dtype=torch.bool).unsqueeze(0).unsqueeze(-1)
        )

        # Keep masks from allowed regions, zero out all others.
        masked_region_values = masks.masked_fill(~allowed, 0.0)

        # Max over regions gives p_{n,k,i}.
        spatial_prior = masked_region_values.max(dim=2).values  # [B, K, N]

        return spatial_prior

    def _prepare_region_masks(self, region_masks: Tensor, num_patches: int) -> Tensor:
        """Convert region masks to shape [B, R, N]."""
        if region_masks.ndim == 4:
            _, num_regions, height, width = region_masks.shape
            if num_regions != self.num_regions:
                raise ValueError(
                    f"Expected {self.num_regions} regions, got {num_regions}."
                )
            if height * width != num_patches:
                raise ValueError(
                    "Flattened region mask grid does not match number of patch tokens: "
                    f"{height} x {width} = {height * width}, but N = {num_patches}."
                )
            region_masks = region_masks.flatten(start_dim=2)

        elif region_masks.ndim == 3:
            _, num_regions, mask_patches = region_masks.shape
            if num_regions != self.num_regions:
                raise ValueError(
                    f"Expected {self.num_regions} regions, got {num_regions}."
                )
            if mask_patches != num_patches:
                raise ValueError(
                    f"region_masks patch dimension must match patch_tokens N={num_patches}, "
                    f"but got {mask_patches}."
                )

        else:
            raise ValueError(
                "region_masks must have shape [B, R, N] or [B, R, H, W], "
                f"got {tuple(region_masks.shape)}."
            )

        return region_masks.to(dtype=torch.float32)

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, num_concepts={self.num_concepts}, "
            f"num_regions={self.num_regions}, attention_dim={self.attention_dim}, "
            f"value_dim={self.value_dim}, hidden_dim={self.hidden_dim}, "
            f"shared_proj_bias={self.shared_proj_bias}, "
            f"prior_strength={self.prior_strength}, eps={self.eps}, "
            f"normalize_queries={self.normalize_queries}, "
            f"freeze_concept_queries={self.freeze_concept_queries}"
        )
