import torch
import torch.nn as nn
from models_.registry import register_model
from utils.concept_utils import load_filtered_concept_region_matrix


def _build_concept_bottleneck_layer(kwargs: dict) -> nn.Module:
    """Shared CBL construction for region-aware concept bottleneck models."""
    from models_.concept_specific_masked_attention import (
        ConceptSpecificMaskedAttentionCBL,
    )

    embed_dim = kwargs.get("embed_dim", 768)
    num_regions = kwargs.get("num_regions", 39)
    concept_names = kwargs.get("concept_names")
    if concept_names is None:
        raise ValueError("concept_names must be provided for region-aware CBL models.")
    concept_names = list(concept_names)
    num_concepts = len(concept_names)

    matrix_path = kwargs.get("concept_region_matrix_path")
    concept_region_matrix = load_filtered_concept_region_matrix(
        matrix_path,
        concept_names,
    )

    query_init = kwargs.get("query_init")
    if query_init is None and kwargs.get("encode_query_init", False):
        from utils.concept_utils import encode_clip_text_embeddings

        clip_model_id = kwargs.get("clip_model", "openai/clip-vit-large-patch14")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        query_init = encode_clip_text_embeddings(
            concept_names,
            clip_model_id=clip_model_id,
            device=device,
        )

    return ConceptSpecificMaskedAttentionCBL(
        input_dim=embed_dim,
        num_concepts=num_concepts,
        num_regions=num_regions,
        concept_region_matrix=concept_region_matrix,
        query_init=query_init,
        attention_dim=kwargs.get("attention_dim"),
        value_dim=kwargs.get("value_dim"),
        hidden_dim=kwargs.get("hidden_dim"),
        shared_proj_bias=bool(kwargs.get("shared_proj_bias", False)),
        prior_strength=float(kwargs.get("prior_strength", 1.0)),
        eps=float(kwargs.get("eps", 1e-6)),
        normalize_queries=bool(kwargs.get("normalize_queries", False)),
        dropout=float(kwargs.get("dropout", 0.0)),
        classifier_bias_init=kwargs.get("classifier_bias_init"),
        freeze_concept_queries=bool(kwargs.get("freeze_concept_queries", False)),
    )


@register_model("RegionAwareCBM")
class RegionAwareCBM(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.embed_dim = kwargs.get("embed_dim", 768)
        self.num_regions = kwargs.get("num_regions", 39)
        num_classes = kwargs.get("num_classes", 2)
        concept_names = kwargs.get("concept_names")
        self.concept_names = list(concept_names)
        self.num_concepts = len(self.concept_names)

        self.concept_bottleneck = _build_concept_bottleneck_layer(kwargs)
        self.classifier = nn.Linear(self.num_concepts, num_classes)

    def forward(self, patch_tokens: torch.Tensor, region_masks: torch.Tensor):
        """
        Args:
            patch_tokens:  (B, 256, embed_dim)
            region_masks:  (B, num_regions, 256)
        Returns:
            class_logits:   (B, num_classes)
            concept_logits: (B, num_concepts)
        """
        concept_logits = self.concept_bottleneck(patch_tokens, region_masks)

        concept_probs = torch.sigmoid(concept_logits)
        class_logits = self.classifier(concept_probs)

        return class_logits, concept_logits
