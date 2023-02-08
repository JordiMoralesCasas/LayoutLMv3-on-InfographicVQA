import os
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from functools import partial
from tqdm import tqdm
import random
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
import logging
from datasets import load_from_disk, DatasetDict, concatenate_datasets, Dataset
from transformers import PreTrainedModel, AutoModelForQuestionAnswering, AutoTokenizer, AutoFeatureExtractor, LayoutLMv3Model, LayoutLMv3Config, BartModel
from modelling.utils import get_optimizers, create_and_fill_np_array, write_data, anls_metric_str, postprocess_qa_predictions, bbox_string
from modelling.tokenization import tokenize_dataset
from modelling.data_collator import DocVQACollator
from modelling.layoutlmv3_gen import LayoutLMv3ForConditionalGeneration


#ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
# not adding find unused because all are used 
accelerator = Accelerator(kwargs_handlers=[])

tqdm = partial(tqdm, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', disable=not accelerator.is_local_main_process)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_file', default="data/cached_datasets/docvqa_cached_extractive_all_lowercase_True_extraction_v1_enumeration", type=str)
    parser.add_argument("--model_folder", default="layoutlmv3-extractive-uncased", type=str)
    parser.add_argument("--load_state", default=None, type=str)

    parser.add_argument("--mode", default="train", type=str, choices=["train", "val", "test"])
    parser.add_argument("--train_batch_size", default=4, type=int, help="Total batch size for training.")
    parser.add_argument("--val_batch_size", default=8, type=int, help="Total batch size for validation.")
    parser.add_argument("--test_batch_size", default=8, type=int, help="Total batch size for test.")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of workers.")
    parser.add_argument("--warmup_step", default=0, type=float, help="Number of warmup steps. When 0, the lr decreses linearly to 0")
    parser.add_argument("--learning_rate", default=5e-6, type=float, help="The peak learning rate.")
    parser.add_argument("--num_epochs", default=1, type=int, help="Number of epochs during training.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='gradient clipping max norm')
    parser.add_argument('--fp16', default=True, action='store_true', help="Whether to use 16-bit 32-bit training")
    parser.add_argument('--debug_run', default=False, action='store_true', help="Run model with 100 samples. For debugging purposes.")
    parser.add_argument('--save_model', default=False, action='store_true', help="Save the model after training")

    parser.add_argument('--use_generation', default=0, type=int, choices=[0, 1], help="Whether to use generation to perform experiments")

    parser.add_argument('--pretrained_model_name', default='microsoft/layoutlmv3-base', type=str, help="pretrained model name")
    parser.add_argument('--stride', default=0, type=int, help="document stride for sliding window, >0 means sliding window, overlapping window")
    parser.add_argument('--ignore_unmatched_span', default=1, type=int, help="ignore unmatched span during training, if not ignored, we treat CLS as the start/end.")

    parser.add_argument('--extraction_nbest', default=20, type=int, help="The nbest span to compare with the ground truth during extraction")
    parser.add_argument('--max_answer_length', default=100, type=int,  help="The maximum answer length")
    args = parser.parse_args()
    for k in args.__dict__:
        logger.info(k + ": " + str(args.__dict__[k]))
    return args


def train(args,
          tokenizer: AutoTokenizer,
          model: PreTrainedModel,
          train_dataloader: DataLoader,
          num_epochs: int, val_metadata,
          valid_dataloader: DataLoader = None,
          valid_dataset_before_tokenized: Dataset = None
          ):
    t_total = int(len(train_dataloader) * num_epochs)
    # warmup_step = 0, linearly decreses from lr to 0
    optimizer, scheduler = get_optimizers(model=model, learning_rate=args.learning_rate, num_training_steps=t_total,
                                          warmup_step=args.warmup_step, eps=1e-8)
    # Prepare for distributed training
    model, optimizer, train_dataloader, valid_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader,
                                                                               valid_dataloader, scheduler)

    best_anls = -1
    os.makedirs(f"model_files/{args.model_folder}", exist_ok=True)  ## create model files. not raise error if exist
    os.makedirs(f"results", exist_ok=True)  ## create model files. not raise error if exist
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        # https://medium.com/@davidlmorton/increasing-mini-batch-size-without-increasing-memory-6794e10db672
        for iter, batch in tqdm(enumerate(train_dataloader, 1), desc="--training batch", total=len(train_dataloader)):
            with torch.cuda.amp.autocast(enabled=bool(args.fp16)):
                output = model(**batch)
                # Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
                loss = output.loss
            total_loss += loss.item()

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if (iter+1) % (args.train_batch_size/2) == 0:
                #print(iter)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

        accelerator.print(
            f"Finish epoch: {epoch}, loss: {total_loss:.2f}, mean loss: {total_loss / len(train_dataloader):.2f}",
            flush=True)
        if valid_dataloader is not None:
            model.eval()
            anls = evaluate(args=args, tokenizer=tokenizer, valid_dataloader=valid_dataloader, model=model,
                            metadata=val_metadata,
                            res_file=f"results/{args.model_folder}.res.json",
                            err_file=f"results/{args.model_folder}.err.json", valid_dataset_before_tokenized=valid_dataset_before_tokenized)
            if anls > best_anls:
                if args.save_model:
                    accelerator.print(f"[Model Info] Saving the best model... with best ANLS: {anls}")
                    module = model.module if hasattr(model, 'module') else model
                    os.makedirs(f"model_files/{args.model_folder}/", exist_ok=True)
                    torch.save(module.state_dict(), f"model_files/{args.model_folder}/state_dict.pth")
                else:
                    accelerator.print(f"[Model Info] Best model... with best ANLS: {anls}")
                best_anls = anls
        elif args.save_model:
            accelerator.print(f"[Model Info] Saving model at epoch {epoch}...")
            module = model.module if hasattr(model, 'module') else model
            os.makedirs(f"model_files/{args.model_folder}/", exist_ok=True)
            torch.save(module.state_dict(), f"model_files/{args.model_folder}/state_dict.pth")
        accelerator.print("****Epoch Separation****")

    accelerator.print(f"[Model Info] Final model with best ANLS: {best_anls}")
    return model


def evaluate(args, tokenizer: AutoTokenizer, valid_dataloader: DataLoader, model: PreTrainedModel,
             valid_dataset_before_tokenized: Dataset, metadata,
             res_file=None, err_file=None):
    model.eval()
    if args.use_generation:
        all_pred_texts = []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=bool(args.fp16)):
            for index, batch in tqdm(enumerate(valid_dataloader), desc="--validation", total=len(valid_dataloader)):
                assert "decoder_input_ids" not in batch
                assert "labels" not in batch
                generated_ids = model(**batch, is_train=False, return_dict=True, max_length=100, num_beams=1)
                generated_ids = accelerator.pad_across_processes(generated_ids, dim=1, pad_index=tokenizer.pad_token_id, pad_first=False)  ## 1 is pad token id
                generated_ids = accelerator.gather_for_metrics(generated_ids)
                preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for g in generated_ids]
                all_pred_texts.extend(preds)
        prediction_list = all_pred_texts
    else:
        all_start_logits = []
        all_end_logits = []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=bool(args.fp16)):
            for index, batch in tqdm(enumerate(valid_dataloader), desc="--validation", total=len(valid_dataloader)):
                batch.start_positions = None
                batch.end_positions = None
                outputs = model(**batch)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
                start_logits = accelerator.pad_across_processes(start_logits, dim=1, pad_index=-100)
                end_logits = accelerator.pad_across_processes(end_logits, dim=1, pad_index=-100)
                all_start_logits.append(accelerator.gather_for_metrics(start_logits).cpu().numpy())
                all_end_logits.append(accelerator.gather_for_metrics(end_logits).cpu().numpy())

        max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor
        eval_dataset = valid_dataloader.dataset
        # concatenate the numpy array
        start_logits_concat = create_and_fill_np_array(all_start_logits, eval_dataset, max_len)
        end_logits_concat = create_and_fill_np_array(all_end_logits, eval_dataset, max_len)
        # delete the list of numpy arrays
        del all_start_logits
        del all_end_logits

        outputs_numpy = (start_logits_concat, end_logits_concat)
        prediction_dict, prediction_list = postprocess_qa_predictions(dataset_before_tokenized = valid_dataset_before_tokenized,
                                                                    metadata=metadata, predictions=outputs_numpy,
                                                                    n_best_size=args.extraction_nbest, max_answer_length=args.max_answer_length)
        all_pred_texts = [prediction['answer'] for prediction in prediction_list]
        
    truth = [data["original_answer"] for data in valid_dataset_before_tokenized]
    accelerator.print(f"prediction: {all_pred_texts[:10]}")
    accelerator.print(f"gold_answers: {truth[:10]}")
    all_anls, anls = anls_metric_str(predictions=all_pred_texts, gold_labels=truth)
    accelerator.print(f"[Info] Average Normalized Lev.S : {anls} ", flush=True)
    if res_file is not None and accelerator.is_main_process and not args.save_model:
        accelerator.print(f"Writing results to {res_file} and {err_file}")
        write_data(data=prediction_list, file=res_file)
    return anls


def main():
    args = parse_arguments()
    set_seed(args.seed, device_specific=True)

    pretrained_model_name = args.pretrained_model_name
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, use_fast=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name, apply_ocr=False)
    if args.use_generation:
        model = LayoutLMv3ForConditionalGeneration(
            LayoutLMv3Config.from_pretrained(pretrained_model_name, return_dict=True))
        old = BartModel.from_pretrained("facebook/bart-base")
        # State loading and config
        model.layoutlmv3.decoder.load_state_dict(old.decoder.state_dict())
        model.layoutlmv3.encoder.load_state_dict(LayoutLMv3Model.from_pretrained(pretrained_model_name).state_dict())
        model.config.decoder_start_token_id = model.config.eos_token_id
        model.config.is_encoder_decoder = True
        model.config.use_cache = True
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(pretrained_model_name)
    if args.load_state != None:
            checkpoint = torch.load(f"model_files/{args.load_state}/state_dict.pth", map_location="cpu")
            model.load_state_dict(checkpoint, strict=True)

    collator = DocVQACollator(tokenizer, feature_extractor, pretrained_model_name=pretrained_model_name, model=model)
    dataset = load_from_disk(args.dataset_file)

    if args.debug_run:
        # For debugging. Use only 100 samples.
        args.save_model = False
        n_limit = 100
        dataset = DatasetDict({
            "train": dataset["train"].select(range(n_limit)), 
            "val": dataset['val'].select(range(n_limit)), 
            "test": dataset['test'].select(range(n_limit))})

    use_msr = "msr_ocr_True" in args.dataset_file
    dataset_name = "docvqa" if "docvqa" in args.dataset_file else "infographicvqa"

    # Image directory
    image_dir = {
        "train": f"data/{dataset_name}/train", 
        "val": f"data/{dataset_name}/val", 
        "test": f"data/{dataset_name}/test"}
    
    tokenized = dataset.map(tokenize_dataset,
                            fn_kwargs={"tokenizer": tokenizer,
                                       "img_dir": image_dir,
                                       "use_msr_ocr": use_msr, # Maybe pass as parameter in the future
                                       "doc_stride": args.stride,
                                       "dataset": dataset_name, 
                                       "use_generation": args.use_generation,
                                       "ignore_unmatched_answer_span_during_train": bool(args.ignore_unmatched_span)},
                            batched=True, num_proc=4,
                            load_from_cache_file=False,
                            remove_columns=dataset["val"].column_names
                            )
    print(tokenized["train"].column_names)
    accelerator.print(tokenized)
    print(tokenized["train"].column_names)
    print("GENERATION", args.use_generation)
    #return

    if args.mode == "train":
        valid_dataloader = DataLoader(tokenized["val"].remove_columns("metadata"), batch_size=args.val_batch_size,
                                                    collate_fn=collator, num_workers=args.num_workers, shuffle=False)
        #for i in valid_dataloader:
        #    print(i)
        #    return
        
        train_dataloader = DataLoader(tokenized["train"].remove_columns("metadata"), batch_size=2,
                                  shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collator)
        #for i in train_dataloader:
        #    print(i)
        #    return
        valid_dataloader = DataLoader(tokenized["val"].remove_columns("metadata"), batch_size=args.val_batch_size,
                                                    collate_fn=collator, num_workers=args.num_workers, shuffle=False)
        train(args=args,
              tokenizer=tokenizer,
              model=model,
              train_dataloader=train_dataloader,
              num_epochs=args.num_epochs,
              valid_dataloader=valid_dataloader,
              valid_dataset_before_tokenized=dataset["val"],
              val_metadata=tokenized["val"]["metadata"])

    elif args.mode == "val":
        valid_dataloader = DataLoader(tokenized["val"].remove_columns("metadata"), batch_size=args.val_batch_size,
                                                    collate_fn=collator, num_workers=args.num_workers, shuffle=False)
        """for i in valid_dataloader:
            print(i)
            break
        return"""
        model, valid_dataloader = accelerator.prepare(model, valid_dataloader)
        model.eval()
        evaluate(args=args,
                 tokenizer=tokenizer,
                 valid_dataloader=valid_dataloader,
                 model=model,
                 valid_dataset_before_tokenized=dataset["val"],
                 metadata=tokenized["val"]["metadata"],
                 res_file=f"results/{args.model_folder}.res.json",
                 err_file=f"results/{args.model_folder}.err.json")
    
    else:
        test_loader = DataLoader(tokenized["test"].remove_columns("metadata"), batch_size=args.test_batch_size,
                                                    collate_fn=collator, num_workers=args.num_workers, shuffle=False)
        checkpoint = torch.load(f"model_files/{args.model_folder}/state_dict.pth", map_location="cpu")
        model.load_state_dict(checkpoint, strict=True)
        model, test_loader = accelerator.prepare(model, test_loader)
        evaluate(args=args,
                 tokenizer=tokenizer,
                 valid_dataloader=test_loader,
                 model=model,
                 valid_dataset_before_tokenized=dataset["test"],
                 metadata=tokenized["test"]["metadata"],
                 res_file=f"results/{args.model_folder}.res.json",
                 err_file=f"results/{args.model_folder}.err.json")


if __name__ == "__main__":
    main()

