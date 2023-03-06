'''
Run inference on the model using the processor and model from above
'''
from transformers import AutoModelForTokenClassification
from transformers import LayoutLMv3ForTokenClassification
import torch
import os
from datasets import load_from_disk

label_list = ['I-question','I-answer','I-header','B-other']

id2label = {k: v for k,v in enumerate(label_list)}
label2id = {v: k for k,v in enumerate(label_list)}

model = LayoutLMv3ForTokenClassification.from_pretrained(
    "us_dl_model",
    id2label=id2label,
    label2id=label2id)

image_column_name = "image"
text_column_name = "tokens"
boxes_column_name = "boxes"
label_column_name = "labels"


## PREPARING FOR INFERENCE
def prepare_examples(examples):
	images = examples[image_column_name]
	words = examples[text_column_name]
	boxes = examples[boxes_column_name]
	word_labels = examples[label_column_name]
	encoding = processor(images, words, boxes=boxes, word_labels=word_labels,
	               truncation=True, padding="max_length")
	return encoding

IMAGES_DIR = f"dataset/testing_data/images"

## fetch names of all annotation files
ANNOTATIONS_DIR = f"dataset/testing_data/annotations"
annotation_files = os.listdir(ANNOTATIONS_DIR)

## globals to store the final data
ALL_LABELS = []
LAYOUT_ANNOTATIONS = []

funsd_labels = {
    'question':0,
    'answer':1,
    'header':2,
    'other':3
}

import json, tqdm
from PIL import Image

## Read files
for annotation_file in tqdm.tqdm(annotation_files):
    with open(f"{ANNOTATIONS_DIR}/{annotation_file}") as f:
        annotations = json.load(f)['form']

    ## Read the corresponding image
    img = Image.open(f"{IMAGES_DIR}/{annotation_file.replace('json','png')}").convert(mode="RGB")

    ## Get the following data from annotation
    ## TOKENS, BBOXES, LABELS
    BBOXES = []
    LABELS = []
    TOKENS = []
    IDS = []

    obj = {}
    for annotation in annotations:
        BBOXES.append(annotation['box'])
        TOKENS.append(annotation['text'])
        LABELS.append(funsd_labels[annotation['label']])
        IDS.append(annotation['id'])

    ALL_LABELS.extend(LABELS)
    obj['ids'] = IDS
    obj['tokens'] = TOKENS
    obj['boxes'] = BBOXES
    obj['labels'] = LABELS
    obj['image'] = img
    LAYOUT_ANNOTATIONS.append(obj)


from datasets import Dataset
from transformers import AutoProcessor

ALL_LABELS = list(set(ALL_LABELS))
new_layout_dataset = Dataset.from_list(LAYOUT_ANNOTATIONS)
column_names = new_layout_dataset.column_names

if torch.cuda.is_available():
  model.to("cuda")
import numpy as np

with torch.no_grad():
    outputs = model(np.array(image))

logits = outputs.logits
predictions = logits.argmax(-1).squeeze().tolist()
labels = encoding.labels.squeeze().tolist()

def unnormalize_box(bbox, width, height):
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]

token_boxes = encoding.bbox.squeeze().tolist()
width, height = image.size

true_predictions = [model.config.id2label[pred] for pred, label in zip(predictions, labels) if label != - 100]
true_labels = [model.config.id2label[label] for prediction, label in zip(predictions, labels) if label != -100]
true_boxes = [unnormalize_box(box, width, height) for box, label in zip(token_boxes, labels) if label != -100]
