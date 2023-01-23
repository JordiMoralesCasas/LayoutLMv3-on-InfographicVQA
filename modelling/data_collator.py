from typing import Optional, List
import torch.nn as nn
from transformers import LayoutLMv3TokenizerFast, LayoutLMv3FeatureExtractor
from dataclasses import dataclass
from PIL import Image


@dataclass
class DocVQACollator:
    tokenizer: LayoutLMv3TokenizerFast
    feature_extractor: LayoutLMv3FeatureExtractor
    pretrained_model_name: str
    padding: bool = True
    model: Optional[nn.Module] = None

    def __call__(self, batch: List):

        labels = [feature["labels"] for feature in batch] if "labels" in batch[0].keys() else None
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            for feature in batch:
                remainder = [self.tokenizer.pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["label_ids"] = feature["labels"] + remainder
                feature.pop("labels")
        # Get image pixel values
        for feature in batch:
            image = Image.open(feature["image"]).convert("RGB")
            vis_features = self.feature_extractor(images=image, return_tensors='np')["pixel_values"][0]
            feature['pixel_values'] = vis_features.tolist()
            if 'image' in feature:
                feature.pop('image')
        # Apply padding if required
        batch = self.tokenizer.pad(
            batch,
            padding=self.padding,
            pad_to_multiple_of=None,
            return_tensors="pt",
            return_attention_mask=True
        )
        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            batch.pop("start_positions")
            batch.pop("end_positions")
            if "label_ids" in batch:
                decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=batch["label_ids"])
                batch["decoder_input_ids"] = decoder_input_ids
                ## for validation, we don't have "labels
        if "label_ids" in batch:
            ## layoutlmv3 tokenizer issue, they don't allow "labels" key as a list..so we did a small trick
            batch["labels"] = batch.pop("label_ids")
        return batch
