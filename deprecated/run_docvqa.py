#!/usr/bin/env python
# coding=utf-8
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np
from datasets import ClassLabel, load_dataset, load_metric, list_datasets, list_metrics, load_from_disk, DatasetDict, concatenate_datasets, Dataset
import cv2
from torch.utils.data import DataLoader
import transformers
from PIL import Image

#from layoutlmft.data import DataCollatorForKeyValueExtraction
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    LayoutLMv3FeatureExtractor,
    LayoutLMv3TokenizerFast,
    AutoFeatureExtractor,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0")

logger = logging.getLogger(__name__)
#from layoutlmft.data.image_utils import RandomResizedCropAndInterpolationWithTwoPic, pil_loader, Compose

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision import transforms
import torch
import torch.nn as nn

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained feature extractor name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default='funsd', metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )
    segment_level_layout: bool = field(default=True)
    visual_embed: bool = field(default=True)
    data_dir: Optional[str] = field(default=None)
    input_size: int = field(default=224, metadata={"help": "images input size for backbone"})
    second_input_size: int = field(default=112, metadata={"help": "images input size for discrete vae"})
    train_interpolation: str = field(
        default='bicubic', metadata={"help": "Training interpolation (random, bilinear, bicubic)"})
    second_interpolation: str = field(
        default='lanczos', metadata={"help": "Interpolation for discrete vae (random, bilinear, bicubic)"})
    imagenet_default_mean_and_std: bool = field(default=False, metadata={"help": ""})


def main():
    # See all possible arguments in layoutlmft/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.dataset_name == 'docvqa':
        """datasets = load_dataset(
            "examples/question_answering/docvqa_loading_script.py"
        )"""
        datasets = load_from_disk("../Data/cached_datasets/docvqa_cached_extractive_all_lowercase_True_msr_False_extraction_v1_enumeration")
        datasets = DatasetDict({"train": datasets["train"].select(range(100)), "val": datasets['val'].select(range(100)), "test": datasets['test'].select(range(100))})
    else:
        raise NotImplementedError()
    print("⏳ Dataset loaded succesfully:\n")

    if training_args.do_train:
        column_names = datasets["train"].column_names
        features = datasets["train"].features
    else:
        column_names = datasets["test"].column_names
        features = datasets["test"].features

    text_column_name = "words" if "words" in column_names else "tokens"

    # Puede dar problemas
    #label_column_name = (
    #    f"{data_args.task_name}_tags" if f"{data_args.task_name}_tags" in column_names else column_names[1]
    #)
    """label_column_name = "answers"

    remove_columns = column_names

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list
    print(features)
    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)"""

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    """config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        input_size=data_args.input_size,
        use_auth_token=True if model_args.use_auth_token else None,
    )"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        tokenizer_file=None,  # avoid loading from a cached file of the pre-trained model in another machine
        cache_dir=model_args.cache_dir,
        use_fast=True,
        add_prefix_space=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # Testing 
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name if model_args.feature_extractor_name else model_args.model_name_or_path,
        apply_ocr=False
    )
    #feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False)

    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        #config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )



    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    """if data_args.visual_embed:
        imagenet_default_mean_and_std = data_args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        common_transform = Compose([
            # transforms.ColorJitter(0.4, 0.4, 0.4),
            # transforms.RandomHorizontalFlip(p=0.5),
            RandomResizedCropAndInterpolationWithTwoPic(
                size=data_args.input_size, interpolation=data_args.train_interpolation),
        ])

        patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])"""

    """# Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples, augmentation=False):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=False,
            truncation=True,
            return_overflowing_tokens=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        labels = []
        bboxes = []
        images = []
        for batch_index in range(len(tokenized_inputs["input_ids"])):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            org_batch_index = tokenized_inputs["overflow_to_sample_mapping"][batch_index]

            label = examples[label_column_name][org_batch_index]
            bbox = examples["bboxes"][org_batch_index]
            previous_word_idx = None
            label_ids = []
            bbox_inputs = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                    bbox_inputs.append([0, 0, 0, 0])
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                    bbox_inputs.append(bbox[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label_to_id[label[word_idx]] if data_args.label_all_tokens else -100)
                    bbox_inputs.append(bbox[word_idx])
                previous_word_idx = word_idx
            labels.append(label_ids)
            bboxes.append(bbox_inputs)

            if data_args.visual_embed:
                ipath = examples["image_path"][org_batch_index]
                img = pil_loader(ipath)
                for_patches, _ = common_transform(img, augmentation=augmentation)
                patch = patch_transform(for_patches)
                images.append(patch)

        tokenized_inputs["labels"] = labels
        tokenized_inputs["bbox"] = bboxes
        if data_args.visual_embed:
            tokenized_inputs["images"] = images

        return tokenized_inputs

    def tokenize_and_align_labels(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            truncation=True,
            return_offsets_mapping=True,
            padding=False,
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []
        for i, offset in enumerate(offset_mapping):
            answer = answers[i][0]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        if data_args.visual_embed:
                tokenized_inputs["images"] = images
        return inputs

    def fuzzy(s1,s2):
        return (editdistance.eval(s1,s2)/((len(s1)+len(s2))/2)) < 0.2

    #https://github.com/herobd/layoutlmv2/blob/main/docvqa_dataset.py
    def subfinder(words_list, answer_list):  
        matches = []
        start_indices = []
        end_indices = []
        for idx, i in enumerate(range(len(words_list))):
            #if words_list[i] == answer_list[0] and words_list[i:i+len(answer_list)] == answer_list:
            if len(words_list[i:i+len(answer_list)])==len(answer_list) and all(fuzzy(words_list[i+j],answer_list[j]) for j in range(len(answer_list))):
                matches.append(answer_list)
                start_indices.append(idx)
                end_indices.append(idx + len(answer_list) - 1)
        if matches:
          return matches[0], start_indices[0], end_indices[0]
        else:
          return None, 0, 0

    # TODO pillar max length de los parametros de entrada
    def tokenize_and_align_labels(example, max_length=512):
        # take a batch 
        questions = example['question']
        words = [w for w in example['words']] #handles numpy and list
        bboxes = example['bboxes']
        # encode it
        encoding = tokenizer(questions, max_length=max_length, padding=False, truncation=True)
        batch_index=0
        input_ids = encoding.input_ids[batch_index].tolist()
        
        # next, add start_positions and end_positions
        start_positions = []
        end_positions = []
        answers = example['answers']
        #print("Batch index:", batch_index)
        cls_index = input_ids.index(tokenizer.cls_token_id)
        # try to find one of the answers in the context, return first match
        words_example = [word.lower() for word in words]
        for answer in answers:
            match, word_idx_start, word_idx_end = subfinder(words_example, answer.lower().split())
            #if match:
            #  break
            # EXPERIMENT (to account for when OCR context and answer don't perfectly match):
            if not match and len(answer)>1:
                for i in range(len(answer)):
                    # drop the ith character from the answer
                    answer_i = answer[:i] + answer[i+1:]
                    # check if we can find this one in the context
                    match, word_idx_start, word_idx_end = subfinder(words_example, answer_i.lower().split())
                    if match:
                        break
            # END OF EXPERIMENT 
            if match:
                sequence_ids = encoding.sequence_ids(batch_index)
                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1
                
                word_ids = encoding.word_ids(batch_index)[token_start_index:token_end_index+1]

                hit=False
                for id in word_ids:
                    if id == word_idx_start:
                        start_positions.append(token_start_index)
                        hit=True
                        break
                    else:
                        token_start_index += 1

                if not hit:
                    continue

                hit=False
                for id in word_ids[::-1]:
                    if id == word_idx_end:
                        end_positions.append(token_end_index)
                        hit=True
                        break
                    else:
                        token_end_index -= 1

                if not hit:
                    end_positions.append(token_end_index)
                
                #print("Verifying start position and end position:")
                #print("True answer:", answer)
                #start_position = start_positions[-1]
                #end_position = end_positions[-1]
                #reconstructed_answer = tokenizer.decode(encoding.input_ids[batch_index][start_position:end_position+1])
                #print("Reconstructed answer:", reconstructed_answer)
                #print("-----------")
            
            #else:
                #print("Answer not found in context")
                #print("-----------")
                #start_positions.append(cls_index)
                #end_positions.append(cls_index)

        if len(start_positions)==0:
            return None

        ans_i = random.randrange(len(start_positions))

        encoding = {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'token_type_ids': encoding['token_type_ids'],
                'bbox': encoding['bbox'],
                }
        
        encoding['image'] = torch.LongTensor(example['image'].copy())
        encoding['start_position'] = torch.LongTensor([start_positions[ans_i]])
        encoding['end_position'] = torch.LongTensor([end_positions[ans_i]])

        return encoding"""

    def bbox_string(box, width, length):
        return [
            int(1000 * (box[0] / width)),
            int(1000 * (box[1] / length)),
            int(1000 * (box[2] / width)),
            int(1000 * (box[3] / length))
        ]

    def get_subword_start_end(word_start, word_end, subword_idx2word_idx, sequence_ids):
        ## find the separator between the questions and the text
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

    def tokenize_docvqa(examples,
                        tokenizer: LayoutLMv3TokenizerFast,
                        img_dir: Dict[str, str],
                        add_metadata: bool = True,
                        use_msr_ocr: bool = False,
                        use_generation: bool = False,
                        doc_stride: int = 128,
                        ignore_unmatched_answer_span_during_train: bool = True): ## doc stride for sliding window, if 0, means no sliding window.

        features = {"input_ids": [], "image":[], "bbox":[], "start_positions": [], "end_positions":[],  "metadata": []}
        current_split = examples["data_split"][0]
        if use_generation:
            features["labels"] = []
        for idx, (question, image_path, words, layout) in enumerate(zip(examples["question"], examples["image"], examples["words"], examples["layout"])):
            current_metadata = {}
            file = os.path.join(img_dir[examples["data_split"][idx]], image_path)
            # img = Image.open(file).convert("RGB")
            answer_list = examples["processed_answers"][idx] if "processed_answers" in examples else []
            original_answer = examples["original_answer"][idx] if "original_answer" in examples else []
            # image_id = f"{examples['ucsf_document_id'][idx]}_{examples['ucsf_document_page_no'][idx]}"
            if len(words) == 0 and current_split == "train":
                continue
            return_overflowing_tokens = doc_stride>0
            tokenized_res = tokenizer.encode_plus(text=question, text_pair=words, boxes=layout, add_special_tokens=True,
                                                max_length=512, truncation="only_second",
                                                return_offsets_mapping=True, stride=doc_stride,
                                                return_overflowing_tokens=return_overflowing_tokens)
            # sample_mapping = tokenized_res.pop("overflow_to_sample_mapping")
            offset_mapping = tokenized_res.pop("offset_mapping")
            if not return_overflowing_tokens:
                offset_mapping = [offset_mapping]

            if use_generation:
                dummy_boxes = [[[0,0,0,0]] for _ in range(len(original_answer))]
                answer_ids = tokenizer.batch_encode_plus([[ans] for ans in original_answer], boxes = dummy_boxes,
                                                        add_special_tokens=True, max_length=100, truncation="longest_first")["input_ids"]
            else:
                answer_ids = [[0] for _ in range(len(original_answer))]
            if not use_msr_ocr:
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
                        if not use_generation:
                            if answer["start_word_position"] == -1:
                                subword_start = 0 ## just use the CLS
                                subword_end = 0
                                num_question_tokens = 0
                                if ignore_unmatched_answer_span_during_train:
                                    continue
                            else:
                                subword_start, subword_end, num_question_tokens = get_subword_start_end(answer["start_word_position"], answer["end_word_position"], subword_idx2word_idx, sequence_ids)
                                if subword_start == -1:
                                    subword_start = 0  ## just use the CLS
                                    subword_end = 0
                                    if ignore_unmatched_answer_span_during_train:
                                        continue
                                if subword_end == -1:
                                    ## that means the end position is out of maximum boundary
                                    ## last is </s>, second last
                                    subword_end = 511 - 1
                        else:
                            features["labels"].append(label_ids)
                            subword_start = -1  ## useless as in generation
                            subword_end = -1
                            num_question_tokens = 0

                        features["image"].append(file)
                        features["input_ids"].append(input_ids)
                        boxes_norms = []
                        for box in bboxes:
                            box_norm = box if use_msr_ocr else bbox_string([box[0], box[1], box[2], box[3]], width, height)
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
                        box_norm = box if use_msr_ocr else bbox_string([box[0], box[1], box[2], box[3]], width, height)
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
                    # print(feature["labels"])
                    feature.pop("labels")

            for feature in batch:
                image = Image.open(feature["image"]).convert("RGB")
                vis_features = self.feature_extractor(images=image, return_tensors='np')["pixel_values"][0]
                if "layoutlmv2" in self.pretrained_model_name:
                    feature["image"] = vis_features.tolist()
                else:
                    feature['pixel_values'] = vis_features.tolist()
                    if 'image' in feature:
                        feature.pop('image')

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

    """if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=remove_columns,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_eval:
        validation_name = "test"
        if validation_name not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets[validation_name]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        eval_dataset = eval_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=remove_columns,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_predict:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))
        test_dataset = test_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=remove_columns,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )"""
    
    # Data collator
    """data_collator = DataCollatorForKeyValueExtraction(
        tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        padding=padding,
        max_length=512,
    )"""

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        if data_args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    data_collator = DocVQACollator(tokenizer, feature_extractor, pretrained_model_name="layoutlmv3", model=model)
    image_dir = {"train": "../Data/DocVQA/train", "val": "../Data/DocVQA/val", "test": "../Data/DocVQA/test"}
    use_msr = False
    tokenized = datasets.map(tokenize_docvqa,
                            fn_kwargs={"tokenizer": tokenizer,
                                       "img_dir": image_dir,
                                       "use_msr_ocr": False,
                                       "use_generation": False,
                                       "doc_stride": 0,
                                       "ignore_unmatched_answer_span_during_train": True},
                            batched=True, num_proc=data_args.preprocessing_num_workers,
                            load_from_cache_file=True,  
                            remove_columns=datasets["val"].column_names
                            )
    # Eliminar (no necesario seguramente)
    train_dataloader = DataLoader(tokenized["train"].remove_columns("metadata"), batch_size=8,
                                  shuffle=True, num_workers=5, pin_memory=True, collate_fn=data_collator)
    eval_dataloader = DataLoader(tokenized["val"].remove_columns("metadata"), batch_size=8,
                                                    collate_fn=data_collator, num_workers=5, shuffle=False)
    test_dataloader = DataLoader(tokenized["test"].remove_columns("metadata"), batch_size=8,
                                                    collate_fn=data_collator, num_workers=5, shuffle=False)

    class MyTrainer(Trainer):
        def __init__(self, model, args, train_dataset=None, eval_dataset=None, compute_metrics=None, collate_fn=None, train_batch_size=8, eval_batch_size=8, test_batch_size=8):
            super().__init__(model, args, train_dataset, eval_dataset, compute_metrics)
            self.collate_fn = collate_fn
            self.train_batch_size = train_batch_size
            self.eval_batch_size = eval_batch_size
            self.test_batch_size = test_batch_size
            self.test_dataset = test_dataset

        def get_train_dataloader(self, train_dataset=None):
            return DataLoader(self.train_dataset,batch_size=self.train_batch_size,
                                  shuffle=True, num_workers=5, pin_memory=True, collate_fn=self.collate_fn)
        def get_eval_dataloader(self, eval_dataset=None):
            return DataLoader(self.eval_dataset, batch_size=self.eval_batch_size,
                                  shuffle=False, num_workers=5, collate_fn=self.collate_fn)
        def get_test_dataloader(self, eval_dataset=None):
            return DataLoader(self.test_dataset, batch_size=self.test_batch_size,
                                                    collate_fn=self.collate_fn, num_workers=5, shuffle=False)

    # Initialize our Trainer
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"].remove_columns("metadata") if training_args.do_train else None,
        eval_dataset=tokenized["val"].remove_columns("metadata") if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        collate_fn=data_collator
    )

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(test_dataset)
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        # Save predictions
        output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_test_predictions_file, "w") as writer:
                for prediction in true_predictions:
                    writer.write(" ".join(prediction) + "\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
