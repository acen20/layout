## Import libraries

import json
import os
import tqdm
from PIL import Image
import numpy as np

## STATICS
IMAGES_DIR = "dataset/training_data/images"

## fetch names of all annotation files
ANNOTATIONS_DIR = "dataset/training_data/annotations"
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
		print(annotation['text'])
		LABELS.append(funsd_labels[annotation['label']])
		IDS.append(annotation['id'])

	ALL_LABELS.extend(LABELS)
	obj['ids'] = IDS
	obj['tokens'] = TOKENS
	obj['boxes'] = BBOXES
	obj['labels'] = LABELS
	obj['image'] = img
	LAYOUT_ANNOTATIONS.append(obj)

ALL_LABELS = list(set(ALL_LABELS))

print(ALL_LABELS)

image_column_name = "image"
text_column_name = "tokens"
boxes_column_name = "boxes"
label_column_name = "labels"


## PREPARING THE EXAMPLES FOR TRAINING
def prepare_examples(examples):
	images = examples[image_column_name]
	words = examples[text_column_name]
	boxes = examples[boxes_column_name]
	word_labels = examples[label_column_name]
	encoding = processor(images, words, boxes=boxes, word_labels=word_labels,
	               truncation=True, padding="max_length")
	print(encoding)
	return encoding

from transformers import AutoProcessor
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
from datasets import Dataset

new_layout_dataset = Dataset.from_list(LAYOUT_ANNOTATIONS)
column_names = new_layout_dataset.column_names

# we'll use the Auto API here - it will load LayoutLMv3Processor behind the scenes,
processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

# we need to define custom features for `set_format` (used later on) to work properly
features = Features({
    'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
    'input_ids': Sequence(feature=Value(dtype='int32')),
    'attention_mask': Sequence(Value(dtype='int32')),
    'bbox': Array2D(dtype="int32", shape=(512, 4)),
    'labels': Sequence(feature=Value(dtype='int32')),
})


final_data = new_layout_dataset.map(
prepare_examples,
batched = True,
features = features,
remove_columns = column_names
)

print(final_data)
