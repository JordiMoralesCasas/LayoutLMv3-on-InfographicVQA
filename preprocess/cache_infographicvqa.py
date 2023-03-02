from tqdm import tqdm
from collections import defaultdict
from datasets import DatasetDict, Dataset
from typing import List
import json
import pickle
from PIL import Image
import textdistance as td
from preprocess.utils import anls_metric_str
from preprocess.extract_positions import extract_start_end_index_v1, extract_start_end_index_v2
from modelling.utils import read_data, bbox_string

from datasets import load_from_disk
"""
    Convert the InfographicVQA dataset into dataset cache, which is a dictionary of Dataset objects.
    At the same time, we extract the answer spans.

"""

def convert_infographicvqa_to_cache(train_file, val_file, test_file, 
                            lowercase: bool, read_msr_ocr: bool = False,
                            extraction_method="v1") -> DatasetDict:

    # Experiment: Create dataset with extractive or non-extractive questions
    if False:
        list_file = open("notebooks/pickled_objects/extractive.list", "rb")
        Id_list = pickle.load(list_file)
        list_file.close()

    data_dict = {}
    for file in [train_file, val_file, test_file]:
        new_all_data = defaultdict(list) # Each entry of the data dictionary will be a list
        data = read_data(file)
        split = data["dataset_split"]
        objs = data['data']
        num_answer_span_found = 0
        all_original_accepted = []
        all_extracted_not_clean = []
        msr_data = None
        msr_image_name2width_height = None

        # Loop over every question/answer pair in the dataset
        for obj_idx, obj in tqdm(enumerate(objs), desc="reading {}".format(split), total=len(objs)):
            # Experiment: Create dataset with infographics with differnt proportions
            if True:
                image_path = f"data/infographicvqa/{split}/infographicVQA_{split}_v1.0_images/{obj['image_local_name']}"
                image = Image.open(image_path)
                aspect_ratio = image.width/image.height
                # Very vertical infographics
                # if aspect_ratio > 0.4:
                #     continue

                # Vertical infographics
                # if aspect_ratio < 0.4 or aspect_ratio > 0.6636871690622075:
                #     continue

                # Infographics with similar proportions as documents
                if aspect_ratio < 0.6636871690622075 or aspect_ratio > 0.885544989926049:
                     continue

                # Horizontal infographics
                # if aspect_ratio < 0.885544989926049 or aspect_ratio > 1.3:
                #    continue
                
                # Very horizontal infographics
                # if aspect_ratio < 1.3:
                #     continue

            # Experiment: Create dataset with extractive or non-extractive questions
            if False:
                if obj["questionId"] not in Id_list and split != "test":
                    continue
            
            new_answers = None
            for key in obj.keys():
                if key == "question":
                    new_all_data[key].append(obj[key].lower() if lowercase else obj[key])
                elif key == "answers":
                    answers = obj[key]
                    # list comprehension
                    new_answers = []
                    for ans in answers:
                        new_answers.append(ans.lower() if lowercase else ans)
                    new_answers = list(set(new_answers))
                    new_all_data["original_answer"].append(new_answers)
                else:
                    new_all_data[key].append(obj[key])
            if new_answers is None:
                # this only applies to test set.
                new_all_data["original_answer"].append(["dummy answer"])
            # Get the concrete ocr file for the current obj
            if read_msr_ocr:
                ocr_file = f"data/infographicvqa/{split}/{split}_msr_ocr_results/{obj['ocr_output_file']}"
            else:
                ocr_file = f"data/infographicvqa/{split}/infographicVQA_{split}_v1.0_ocr_outputs/{obj['ocr_output_file']}"
            ocr_data = read_data(ocr_file)

            if not read_msr_ocr: # Sin ocr mrs
                new_all_data["image"].append(f"infographicVQA_{split}_v1.0_images/{obj['ocr_output_file']}")
                if "LINE" in ocr_data.keys():
                    all_text = ' '.join([line['Text'] for line in ocr_data["LINE"]])
                    text, layout = [], []
                    for word in ocr_data["WORD"]:
                        # Define each bbox with 2 points instead of 4
                        BoundingBox = word['Geometry']["BoundingBox"]
                        new_x1 = BoundingBox["Left"]
                        new_x2 = new_x1 + BoundingBox["Width"]
                        new_y2 = BoundingBox["Top"]
                        new_y1 = new_y2 - BoundingBox["Height"]
                        if word["Text"].startswith("http") or word["Text"] == "":
                            continue
                        text.append(word["Text"].lower() if lowercase else word["Text"])
                        layout.append([new_x1, new_y1, new_x2, new_y2])
                else:
                    # If the ocr does not have any text, the text is an empty string and the bounding box covers the whole image
                    BoundingBox = ocr_data["PAGE"][0]['Geometry']["BoundingBox"]
                    new_x1 = BoundingBox["Left"]
                    new_x2 = new_x1 + BoundingBox["Width"]
                    new_y2 = BoundingBox["Top"]
                    new_y1 = new_y2 - BoundingBox["Height"]
                    text = [""]
                    layout = [[new_x1, new_y1, new_x2, new_y2]]
            else:
                if not ocr_data["is_resized"]:
                    new_all_data["image"].append(f"infographicVQA_{split}_v1.0_images/{obj['image_local_name']}")
                else:
                    new_all_data["image"].append(f"resized_images/{obj['image_local_name']}")
                                
                all_text = ' '.join([line['text'] for line in ocr_data['recognitionResults'][0]['lines']])
                text, layout = [], []
                for line in ocr_data["recognitionResults"][0]["lines"]:
                    # Define each bbox with 2 points instead of 4
                    for word in line["words"]:
                        x1, y1, x2, y2, x3, y3, x4, y4 = word['boundingBox']
                        new_x1 = min([x1, x2, x3, x4])
                        new_x2 = max([x1, x2, x3, x4])
                        new_y1 = min([y1, y2, y3, y4])
                        new_y2 = max([y1, y2, y3, y4])
                        if word["text"].startswith("http") or word["text"] == "":
                            continue
                        text.append(word["text"].lower() if lowercase else word["text"])
                        layout.append([new_x1, new_y1, new_x2, new_y2])

            new_all_data['ocr_text'].append(all_text)
            new_all_data['words'].append(text)
            new_all_data['layout'].append(layout)
            if new_answers is not None:
                ## lowercase everything for matching
                before_processed_text = [w.lower() for w in text]
                before_processed_new_answers = [a.lower() for a in new_answers]
                if extraction_method == "v1":
                    processed_answers, all_not_found = extract_start_end_index_v1(before_processed_new_answers, before_processed_text)
                elif extraction_method == "v2":
                    processed_answers, all_not_found = extract_start_end_index_v2(before_processed_new_answers, before_processed_text)
                elif extraction_method == "v1_v2":
                    processed_answers, all_not_found = extract_start_end_index_v1(before_processed_new_answers, before_processed_text)
                    if all_not_found:
                        processed_answers, _ = extract_start_end_index_v2(before_processed_new_answers, before_processed_text)
                elif extraction_method == "v2_v1":
                    processed_answers, all_not_found = extract_start_end_index_v2(before_processed_new_answers, before_processed_text)
                    if all_not_found:
                        processed_answers, _ = extract_start_end_index_v1(before_processed_new_answers, before_processed_text)
            else:
                processed_answers = [{
                    "start_word_position": -1,
                    "end_word_position": -1,
                    "gold_answer": "<NO_GOLD_ANSWER>",
                    "extracted_answer": ""}]
            new_all_data['processed_answers'].append(processed_answers)

            ## Note: just to count the stat
            for ans in processed_answers:
                if ans['start_word_position'] != -1:
                    num_answer_span_found += 1
                    break
            #NOTE: check the current extracted ANLS
            current_extracted_not_clean = []
            for ans in processed_answers:
                if ans['start_word_position'] != -1:
                    current_extracted_not_clean.append(' '.join(text[ans['start_word_position']:ans['end_word_position']+1]))
            if len(current_extracted_not_clean) > 0:
                # _, anls = anls_metric_str(predictions=[current_extracted_not_clean], gold_labels=[new_answers])
                all_extracted_not_clean.append(current_extracted_not_clean)
                all_original_accepted.append(new_answers)
        # NOTE: check all extracted ANLS
        if "test" not in file:
            _, anls = anls_metric_str(predictions=all_extracted_not_clean, gold_labels=all_original_accepted)
            print(f"Current ANLS: {anls}")
        total_num = len(new_all_data["questionId"])
        print(f"{split} has {total_num} questions, "
              f"extractive answer found: {num_answer_span_found} "
              f"answer not found: {total_num - num_answer_span_found}", flush=True)
        data_dict[split] = Dataset.from_dict(new_all_data)
        
    all_data = DatasetDict(data_dict)
    return all_data



if __name__ == '__main__':
    all_lowercase = True
    answer_extraction_methods = ["v1"]
    for answer_extraction_method in answer_extraction_methods:
        for read_msr in [True]:
            print(answer_extraction_method.capitalize(), read_msr)
            dataset = convert_infographicvqa_to_cache(
                                            "data/infographicvqa/train/infographicVQA_train_v1.0.json",
                                            "data/infographicvqa/val/infographicVQA_val_v1.0.json",
                                            "data/infographicvqa/test/infographicVQA_test_v1.0.json",
                                            lowercase=all_lowercase,read_msr_ocr=read_msr,
                                            extraction_method=answer_extraction_method)
            #cached_filename = f"cached_datasets/infographicvqa_very_vertical_msr_ocr_{read_msr}_extraction_{answer_extraction_method}_enumeration"
            #cached_filename = f"cached_datasets/infographicvqa_vertical_msr_ocr_{read_msr}_extraction_{answer_extraction_method}_enumeration"
            #cached_filename = f"cached_datasets/infographicvqa_document_msr_ocr_{read_msr}_extraction_{answer_extraction_method}_enumeration"
            #cached_filename = f"cached_datasets/infographicvqa_horizontal_msr_ocr_{read_msr}_extraction_{answer_extraction_method}_enumeration"
            cached_filename = f"cached_datasets/infographicvqa_very_horizontal_msr_ocr_{read_msr}_extraction_{answer_extraction_method}_enumeration"
            dataset.save_to_disk(cached_filename)     