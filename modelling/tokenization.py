from typing import Dict
import os
from transformers import LayoutLMv3TokenizerFast
from PIL import Image
import cv2
from modelling.utils import bbox_string

def get_subword_start_end(word_start, word_end, subword_idx2word_idx, sequence_ids):
    ## find the separator between the questions and the text

    # Look for the start of the context
    start_of_context = -1
    for i in range(len(sequence_ids)):
        if sequence_ids[i] == 1:
            start_of_context = i
            break
    num_question_tokens = start_of_context
    assert start_of_context != -1, "Could not find the start of the context"
    subword_start = -1
    subword_end = -1
    for i in range(start_of_context, len(subword_idx2word_idx)):
        if word_start == subword_idx2word_idx[i] and subword_start == -1:
            subword_start = i
        if word_end == subword_idx2word_idx[i]:
            subword_end = i
    return subword_start, subword_end, num_question_tokens

"""
handling the out of maximum length reference:
https://github.com/huggingface/notebooks/blob/main/examples/question_answering.ipynb
"""

def tokenize_dataset(examples,
                    tokenizer: LayoutLMv3TokenizerFast,
                    img_dir: Dict[str, str],
                    add_metadata: bool = True,
                    use_msr_ocr: bool = False,
                    dataset: str = "docvqa",
                    doc_stride: int = 128,
                    ignore_unmatched_answer_span_during_train: bool = True): ## doc stride for sliding window, if 0, means no sliding window.

    features = {"input_ids": [], "image":[], "bbox":[], "start_positions": [], "end_positions":[],  "metadata": []}
    current_split = examples["data_split"][0]
    for idx, (question, image_path, words, layout) in enumerate(zip(examples["question"], examples["image"], examples["words"], examples["layout"])):
        current_metadata = {}
        file = os.path.join(img_dir[examples["data_split"][idx]], image_path)
        answer_list = examples["processed_answers"][idx] if "processed_answers" in examples else []
        original_answer = examples["original_answer"][idx] if "original_answer" in examples else []
        if len(words) == 0 and current_split == "train":
            continue
        return_overflowing_tokens = doc_stride > 0
        # Tokenize text
        tokenized_res = tokenizer(text=question, text_pair=words, boxes=layout, add_special_tokens=True,
                                              max_length=512, truncation="only_second",
                                              return_offsets_mapping=True, stride=doc_stride,
                                              return_overflowing_tokens=return_overflowing_tokens)
        offset_mapping = tokenized_res.pop("offset_mapping")
        if not return_overflowing_tokens:
            offset_mapping = [offset_mapping]

        answer_ids = [[0] for _ in range(len(original_answer))]
        if not (use_msr_ocr and dataset == "docvqa"):
            img = cv2.imread(file)
            height, width = img.shape[:2]

        for stride_idx, offsets in enumerate(offset_mapping):
            input_ids = tokenized_res["input_ids"][stride_idx] if return_overflowing_tokens else tokenized_res["input_ids"]
            bboxes = tokenized_res["bbox"][stride_idx] if return_overflowing_tokens else tokenized_res["bbox"]
            subword_idx2word_idx = tokenized_res.encodings[stride_idx].word_ids
            sequence_ids = tokenized_res.encodings[stride_idx].sequence_ids
            if current_split == "train":
                # for training, we treat instances with multiple answers as multiple instances
                for answer, label_ids in zip(answer_list, answer_ids):
                    if answer["start_word_position"] == -1:
                        subword_start = 0 ## just use the CLS
                        subword_end = 0
                        num_question_tokens = 0
                        if ignore_unmatched_answer_span_during_train:
                            continue
                    else:
                        subword_start, subword_end, num_question_tokens = get_subword_start_end(answer["start_word_position"], answer["end_word_position"], subword_idx2word_idx, sequence_ids)
                        if subword_start == -1:
                            subword_start = 0 ## just use the CLS
                            subword_end = 0
                            if ignore_unmatched_answer_span_during_train:
                                continue
                        if subword_end == -1:
                            ## that means the end position is out of maximum boundary
                            ## last is </s>, second last
                            subword_end = 511 - 1

                    features["image"].append(file)
                    features["input_ids"].append(input_ids)
                    boxes_norms = []
                    for box in bboxes:
                        box_norm = box if use_msr_ocr and dataset == "docvqa" else bbox_string([box[0], box[1], box[2], box[3]], width, height)
                        boxes_norms.append(box_norm)
                    features["bbox"].append(boxes_norms)
                    features["start_positions"].append(subword_start)
                    features["end_positions"].append(subword_end)
                    current_metadata["original_answer"] = original_answer
                    current_metadata["question"] = question
                    current_metadata["num_question_tokens"] = num_question_tokens ## only used in testing.
                    current_metadata["words"] = words
                    current_metadata["subword_idx2word_idx"] = subword_idx2word_idx
                    current_metadata["questionId"] = examples["questionId"][idx]
                    current_metadata["data_split"] = examples["data_split"][idx]
                    features["metadata"].append(current_metadata)
                    if not add_metadata:
                        features.pop("metadata")
            else:
                # for validation and test, we treat instances with multiple answers as one instance
                # we just use the first one, and put all the others in the "metadata" field
                subword_start, subword_end = -1, -1
                for i in range(len(sequence_ids)):
                    if sequence_ids[i] == 1:
                        num_question_tokens = i
                        break
                features["image"].append(file)
                features["input_ids"].append(input_ids)
                boxes_norms = []
                for box in bboxes:
                    box_norm = box if use_msr_ocr and dataset == "docvqa" else bbox_string([box[0], box[1], box[2], box[3]], width, height)
                    boxes_norms.append(box_norm)
                features["bbox"].append(boxes_norms)
                features["start_positions"].append(subword_start)
                features["end_positions"].append(subword_end)
                current_metadata["original_answer"] = original_answer
                current_metadata["question"] = question
                current_metadata["num_question_tokens"] = num_question_tokens
                current_metadata["words"] = words
                current_metadata["subword_idx2word_idx"] = subword_idx2word_idx
                current_metadata["questionId"] = examples["questionId"][idx]
                current_metadata["data_split"] = examples["data_split"][idx]
                features["metadata"].append(current_metadata)
                if not add_metadata:
                    features.pop("metadata")
    return features
