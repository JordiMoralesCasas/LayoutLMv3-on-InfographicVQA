from typing import Optional, List
import torch.nn as nn
from transformers import LayoutLMv3TokenizerFast, LayoutLMv3FeatureExtractor
from dataclasses import dataclass
from PIL import Image, ImageOps
import numpy as np



def pad_images(batch, add_pixel_mask=False):
    w_list, h_list = [], []
    for element in batch:
        h_list.append(len(element["pixel_values"][0]))
        w_list.append(len(element["pixel_values"][0][0]))
    w_final = max(w_list)
    h_final = max(h_list)
    for i, element in enumerate(batch):
        w, h = w_list[i], h_list[i]
        w_diff, h_diff = w_final - w, h_final - h
        # Do same padding for each channel
        for channel in range(3):
            # Add as many padded rows as necessary
            for j in range(h_diff):
                element["pixel_values"][channel].append([0]*w_final)
            # Add padded columns
            for row in element["pixel_values"][channel]:
                row += [0]*w_diff
        if add_pixel_mask:
            p_mask = np.zeros((h_final, w_final))
            p_mask[:h, :w] = 1
            element["pixel_mask"] = p_mask.tolist()

@dataclass
class DocVQACollator:
    tokenizer: LayoutLMv3TokenizerFast
    feature_extractor: LayoutLMv3FeatureExtractor
    pretrained_model_name: str
    padding: bool = True
    resize: bool = True
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
            # Experiment: Increase the ammount of patches
            if not self.resize:
                standard_size = self.model.config.input_size # TODO: get from config or parameter for the class(?)
                patch_size = self.model.config.patch_size
                max_horizontal_patches = self.model.config.max_horizontal_patches
                max_vertical_patches = self.model.config.max_vertical_patches
                if image.width < image.height:
                    large_size = int((image.height*(standard_size/image.width)) // patch_size)
                    image = image.resize((
                        standard_size, 
                        large_size*patch_size if large_size < max_vertical_patches else max_vertical_patches*patch_size))
                else:
                    large_size = int((image.width*(standard_size/image.height)) // patch_size)
                    image = image.resize((
                        large_size*patch_size if large_size < max_horizontal_patches else max_horizontal_patches*patch_size, 
                        standard_size))     

            vis_features = self.feature_extractor(images=image, return_tensors='np')["pixel_values"][0]
            feature['pixel_values'] = vis_features.tolist()
            if 'image' in feature:
                feature.pop('image')

        # Pad images and create their pixel masks
        pad_images(
            batch, 
            add_pixel_mask=True
        )

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
                ## for validation, we don't have "labels" since they are only used for calculing the loss
        if "label_ids" in batch:
            ## layoutlmv3 tokenizer issue, they don't allow "labels" key as a list..so we did a small trick
            batch["labels"] = batch.pop("label_ids")
        return batch
