# coding=utf-8
'''
Reference: https://huggingface.co/datasets/nielsr/funsd/blob/main/funsd.py
'''
import json
import os

import datasets

from layoutlmft.data.image_utils import load_image, normalize_bbox


logger = datasets.logging.get_logger(__name__)


_CITATION = """"""

_DESCRIPTION = """ TODO longer description\
https://www.docvqa.org/
"""

class DocVQAConfig(datasets.BuilderConfig):
    """BuilderConfig for FUNSD"""

    def __init__(self, **kwargs):
        """BuilderConfig for FUNSD.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(DocVQAConfig, self).__init__(**kwargs)


class DocVQA(datasets.GeneratorBasedBuilder):
    """TODO add description"""

    VERSION = datasets.Version("1.0.0")

    # Mirar mejor esto
    BUILDER_CONFIGS = [
        DocVQAConfig(name="docvqa", version=VERSION, description="DocVQA dataset"),
    ]

    def _info(self):
        features = datasets.Features(
                {
                    "questionId": datasets.Value("int32"),
                    "question": datasets.Value("string"),
                    "docId": datasets.Value("int32"),
                    "ucsf_document_id": datasets.Value("string"),
                    "ucsf_document_page_no": datasets.Value("string"),
                    "answers": datasets.Sequence(datasets.Value("string")),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                    "image_path": datasets.Value("string"),
                    "words": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                }
            )
        
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage="https://www.docvqa.org/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO use relative path
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": f"/home/jmorales/tfg/source/Data/DocVQA/train",
                    "split": "train"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": f"/home/jmorales/tfg/source/Data/DocVQA/test",
                    "split": "test"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": f"/home/jmorales/tfg/source/Data/DocVQA/val",
                    "split": "val"
                }
            ),
        ]

    def _generate_examples(self, filepath, split):
        logger.info("⏳ Generating examples from = %s", filepath)
        img_dir = os.path.join(filepath, "documents")
        ocr_dir = os.path.join(filepath, "ocr_results")
        # JSON file with the question-answer pairs
        split_data = os.path.join(filepath, f"{split}_v1.0.json")
        with open(split_data, 'r') as f:
            # Access to the data list
            data = json.load(f)["data"]

        for key, question_data in enumerate(data):
            questionId = question_data["questionId"]
            question = question_data["question"]
            docId = question_data["docId"]
            ucsf_document_id = question_data["ucsf_document_id"]
            ucsf_document_page_no = question_data["ucsf_document_page_no"]
            answers = question_data["answers"] if split != "test" else None,

            image_path = os.path.join(filepath, question_data["image"])
            image, size = load_image(image_path)
            # ocr data
            ocr_words = []
            bboxes = []
            line_bboxes = []
            ocr_path = os.path.join(
                ocr_dir,
                f"{ucsf_document_id}_{ucsf_document_page_no}.json"
            )
            with open(ocr_path, 'r') as f:
                # Access to the ocr info
                ocr_data = json.load(f)["recognitionResults"]

            for page_data in ocr_data:
                for line_data in page_data["lines"]:
                    line_bboxes.append(line_data["boundingBox"])
                    for word_data in line_data["words"]:
                        ocr_words.append(word_data["text"])
                        bboxes.append(word_data["boundingBox"])
            # by default: --segment_level_layout 1
            # if do not want to use segment_level_layout, comment the following line
            bboxes.extend(line_bboxes)

            """for item in ocr_data:
                cur_line_bboxes = []
                words, label = item["words"], item["label"]
                words = [w for w in words if w["text"].strip() != ""]
                if len(words) == 0:
                    continue
                if label == "other":
                    for w in words:
                        tokens.append(w["text"])
                        ner_tags.append("O")
                        cur_line_bboxes.append(normalize_bbox(w["box"], size))
                else:
                    tokens.append(words[0]["text"])
                    ner_tags.append("B-" + label.upper())
                    cur_line_bboxes.append(normalize_bbox(words[0]["box"], size))
                    for w in words[1:]:
                        tokens.append(w["text"])
                        ner_tags.append("I-" + label.upper())
                        cur_line_bboxes.append(normalize_bbox(w["box"], size))"""
            #print(key, questionId, question, docId, ucsf_document_id, ucsf_document_page_no, answers, image, image_path, ocr_words, bboxes)
            yield key, {
                "questionId": questionId,
                "question": question,
                "docId": docId,
                "ucsf_document_id": ucsf_document_id,
                "ucsf_document_page_no": ucsf_document_page_no,
                "answers": answers,
                "image": image,
                "image_path": image_path,
                "words": ocr_words,
                "bboxes": bboxes,
            }