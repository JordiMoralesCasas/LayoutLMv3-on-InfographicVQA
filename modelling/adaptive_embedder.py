from typing import Optional, Tuple, Union
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    QuestionAnsweringModelOutput,
    BaseModelOutput
)
from transformers import LayoutLMv3Model, LayoutLMv3ForQuestionAnswering
from transformers.models.layoutlmv3.modeling_layoutlmv3 import LayoutLMv3ClassificationHead
from timm.models.layers import to_2tuple

from torch.nn import CrossEntropyLoss

class LayoutLMv3ModelNewEmbeddings(LayoutLMv3Model):
    """
    """

    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config):
        super().__init__(config)

        self.patch_size = config.patch_size
        self.hidden_size = config.hidden_size
        self.max_horizontal_patches = config.max_horizontal_patches 
        self.max_vertical_patches = config.max_vertical_patches 
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.pad_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        
        # Positional 2D embedding parameters
        self.pos_embed_X = nn.Parameter(torch.zeros(1, 1, self.max_horizontal_patches, self.hidden_size))
        self.pos_embed_Y = nn.Parameter(torch.zeros(1, self.max_vertical_patches, 1, self.hidden_size))


    def forward_image(self, device, pixel_values, visual_attention_mask):
        embeddings = self.patch_embed(pixel_values)
        
        patch_grid = (
            int(pixel_values.size(dim=2) / self.patch_size), 
            int(pixel_values.size(dim=3) / self.patch_size))
        batch_size, seq_len, _ = embeddings.size()

        # Treat padding tokens as masked tokens
        pad_tokens = self.pad_token.expand(batch_size, seq_len, -1)
        mask = visual_attention_mask.unsqueeze(-1).type_as(pad_tokens)
        embeddings = embeddings * mask + pad_tokens * (1.0 - mask)

        # Apply the 2D positional embedding (only to the non-padded tokens)
        # Loop over each document in the batch
        for i in range(batch_size):
            # Patches that are not padding
            non_pad_grid = (
                int(torch.sum(visual_attention_mask[i, ::patch_grid[1]])),
                int(torch.sum(visual_attention_mask[i, :patch_grid[1]]))
            )
            # For training all the feature map more uniformly, we randomly apply a translation.
            # First choose if they will be translated vertically or horizontally (the probability is given by the
            # ratio between the vertical and horizontal maximums)
            # Then, randomly choose how many patches they will be translated.
            phase_v = phase_h = 0
            if self.training:
                if torch.bernoulli(torch.tensor([self.max_horizontal_patches/self.max_vertical_patches])) == 1:
                    # Vertical translation
                    diff = int(self.max_vertical_patches - non_pad_grid[0])
                    phase_v =  0 if diff == 0 else int(torch.randint(0, diff, [1]))
                else:
                    # Horizontal translation
                    diff = int(self.max_horizontal_patches - non_pad_grid[1])
                    phase_h = 0 if diff == 0 else  int(torch.randint(0, diff, [1]))
                
            expand_shape = (*non_pad_grid, self.hidden_size)
            reshape_shape = (patch_grid[0]*patch_grid[1], self.hidden_size)

            pos_embed_X = self.pos_embed_X[0, :, phase_h:phase_h+non_pad_grid[1], :].expand(expand_shape)
            pos_embed_Y = self.pos_embed_Y[0, phase_v:phase_v+non_pad_grid[0], :, :].expand(expand_shape)

            pos_embed = torch.zeros((*patch_grid, self.hidden_size))
            pos_embed[:non_pad_grid[0], :non_pad_grid[1], :] = pos_embed_X + pos_embed_Y
            pos_embed = pos_embed.reshape(reshape_shape).to(device)
            embeddings = embeddings + pos_embed

        # Add [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        embeddings = self.pos_drop(embeddings)
        embeddings = self.norm(embeddings)

        return embeddings

    def calculate_visual_bbox(self, device, dtype, batch_size, patch_grid, visual_attention_mask, max_len=1000):
        for i in range(batch_size):
            non_pad_grid = (
                int(torch.sum(visual_attention_mask[i, ::patch_grid[1]])),
                int(torch.sum(visual_attention_mask[i, :patch_grid[1]]))
            )
            # The bboxes for each token (not padding) go from 0 to max_len
            visual_bbox_x = torch.div(
                torch.arange(0, max_len * (patch_grid[1] + 1), max_len), non_pad_grid[1], rounding_mode="trunc"
            )
            visual_bbox_y = torch.div(
                torch.arange(0, max_len * (patch_grid[0] + 1), max_len), non_pad_grid[0], rounding_mode="trunc"
            )
            visual_bbox_i = torch.stack(
                [
                    visual_bbox_x[:-1].repeat(patch_grid[0], 1),
                    visual_bbox_y[:-1].repeat(patch_grid[1], 1).transpose(0, 1),
                    visual_bbox_x[1:].repeat(patch_grid[0], 1),
                    visual_bbox_y[1:].repeat(patch_grid[1], 1).transpose(0, 1),
                ],
                dim=-1,
            ).view(-1, 4)

            cls_token_box = torch.tensor([[0 + 1, 0 + 1, max_len - 1, max_len - 1]])
            visual_bbox_i = torch.cat([cls_token_box, visual_bbox_i], dim=0).unsqueeze(0)
            if i == 0:
                visual_bbox = visual_bbox_i
                
            else:
                visual_bbox = torch.cat((visual_bbox, visual_bbox_i))

        # Use [0,0,0,0] as the bounding boxes for the padding tokens
        visual_bbox[~visual_attention_mask.type(torch.bool)] = torch.tensor([0, 0, 0, 0])

        visual_bbox = visual_bbox.to(device).type(dtype)
        return visual_bbox

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, BaseModelOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
            device = inputs_embeds.device
        elif pixel_values is not None:
            batch_size = len(pixel_values)
            device = pixel_values.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds or pixel_values")

        if input_ids is not None or inputs_embeds is not None:
            if attention_mask is None:
                attention_mask = torch.ones(((batch_size, seq_length)), device=device)
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            if bbox is None:
                bbox = torch.zeros(tuple(list(input_shape) + [4]), dtype=torch.long, device=device)

            text_embedding = self.embeddings(
                input_ids=input_ids,
                bbox=bbox,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
            )
        final_bbox = final_position_ids = None
        patch_grid = (None, None)
        if pixel_values is not None:
            patch_size = self.config.patch_size
            patch_grid = int(pixel_values.shape[2] / patch_size), int(
                pixel_values.shape[3] / patch_size
            )

            # Padded patches from the pixel mask
            visual_attention_mask = pixel_mask[:, ::patch_size, ::patch_size].flatten(1)
            visual_embeddings = self.forward_image(device, pixel_values, visual_attention_mask)

            # Increase attention mask to include the CLS token of the visual embedding
            visual_attention_mask = torch.cat([
                torch.ones((batch_size, 1), dtype=torch.long, device=device), 
                visual_attention_mask
                ], 1)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)
            else:
                attention_mask = visual_attention_mask

            if self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
                if self.config.has_spatial_attention_bias:
                    visual_bbox = self.calculate_visual_bbox(
                        device, dtype=torch.long, 
                        batch_size=batch_size, 
                        patch_grid=patch_grid,
                        visual_attention_mask=visual_attention_mask
                        )
                    if bbox is not None:
                        final_bbox = torch.cat([bbox, visual_bbox], dim=1)
                    else:
                        final_bbox = visual_bbox

                visual_position_ids = torch.arange(
                    0, visual_embeddings.shape[1], dtype=torch.long, device=device
                ).repeat(batch_size, 1)
                if input_ids is not None or inputs_embeds is not None:
                    position_ids = torch.arange(0, input_shape[1], device=device).unsqueeze(0)
                    position_ids = position_ids.expand(input_shape)
                    final_position_ids = torch.cat([position_ids, visual_position_ids], dim=1)
                else:
                    final_position_ids = visual_position_ids

            if input_ids is not None or inputs_embeds is not None:
                embedding_output = torch.cat([text_embedding, visual_embeddings], dim=1)
            else:
                embedding_output = visual_embeddings

            embedding_output = self.LayerNorm(embedding_output)
            embedding_output = self.dropout(embedding_output)

            """print(visual_embeddings.size(), "visual embedding")
            print(text_embedding.size(), "text embedding")
            print(embedding_output.size(), "final embedding")
            print(visual_bbox.size(), "visual bbox")
            print(bbox.size(), "text bbox")
            print(final_bbox.size(), "final bbox")"""

        elif self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
            if self.config.has_spatial_attention_bias:
                final_bbox = bbox
            if self.config.has_relative_attention_bias:
                position_ids = self.embeddings.position_ids[:, : input_shape[1]]
                position_ids = position_ids.expand_as(input_ids)
                final_position_ids = position_ids

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, None, device, dtype=embedding_output.dtype
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            bbox=final_bbox,
            position_ids=final_position_ids,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            patch_height=patch_grid[0],
            patch_width=patch_grid[1],
        )

        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class LayoutLMv3ForQuestionAnsweringNewEmbeddings(LayoutLMv3ForQuestionAnswering):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.layoutlmv3 = LayoutLMv3ModelNewEmbeddings(config)

    # TODO: Quizas no haga falta tocar este forward
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        bbox: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.LongTensor] = None,
        pixel_mask: Optional[torch.FloatTensor] = None
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv3(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            bbox=bbox,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )