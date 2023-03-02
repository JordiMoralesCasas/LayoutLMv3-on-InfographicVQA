from typing import Optional, List
import torch.nn as nn
from transformers import LayoutLMv3TokenizerFast, LayoutLMv3FeatureExtractor
from dataclasses import dataclass
from PIL import Image, ImageOps
import numpy as np

@dataclass
class DocVQACollator:
    tokenizer: LayoutLMv3TokenizerFast
    feature_extractor: LayoutLMv3FeatureExtractor
    pretrained_model_name: str
    padding: bool = True
    resize: bool = True
    multiple_embeddings: bool = False
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
                
            # Experiment: Create new docvqa with documents of varying proportions
            if "image_mod" in feature:
                # Artificially create new documents with different proportions, depending on the "image_mod" columns.
                # V1: Normal, horizontally, verically
                if False:
                    if feature["image_mod"] == 2:
                        # Stretch horizontally
                        ratio = 1.5
                        image = image.resize(
                            (
                                int(np.floor(image.height*1.5)), 
                                image.height
                            ))
                    elif feature["image_mod"] == 3:
                        # Stretch vertically
                        ratio = 0.3
                        image = image.resize(
                            (
                                image.width, 
                                int(np.floor(image.width*(1/ratio)))
                            ))
                    # else: "image_type" == 1, maintain the original aspect ratio
                # V2: Normal, vertically (a bit), vertically (a bunch)
                else:
                    if feature["image_mod"] == 2:
                        # Stretch vertically (not much)
                        ratio = 0.5
                        image = image.resize(
                            (
                                image.width, 
                                int(np.floor(image.width*(1/ratio)))
                            ))
                    elif feature["image_mod"] == 3:
                        # Stretch vertically
                        ratio = 0.2
                        image = image.resize(
                            (
                                image.width, 
                                int(np.floor(image.width*(1/ratio)))
                            ))
                    # else: "image_type" == 1, maintain the original aspect ratio
            
            # Keep track of the aspect ratio of the image if we are using the multiple embeddings approach
            if self.multiple_embeddings:
                feature["aspect_ratio"] = image.width / image.height
            
            # Stop feature extractor from resizing the images.
            # Instead add black stripes to keep the original proportions
            if not self.resize:
                size = image.size
                stripe_width = int(np.floor((max(size) - min(size))/2))
                # border = (left, top, right, bottom) 
                border = (0, stripe_width, 0, stripe_width) if size[0] > size[1] else (stripe_width, 0, stripe_width, 0)  
                image = ImageOps.expand(image, border=border, fill="black")

            vis_features = self.feature_extractor(images=image, return_tensors='np')["pixel_values"][0]
            feature['pixel_values'] = vis_features.tolist()
            if 'image' in feature:
                feature.pop('image')
            if 'image_mod' in feature:
                feature.pop('image_mod')
        # Apply padding if required
        batch = self.tokenizer.pad(
            batch,
            padding=self.padding,
            pad_to_multiple_of=None,
            return_tensors="pt",
            return_attention_mask=True
        )
        # FOR GENERATION: prepare decoder_input_ids
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
