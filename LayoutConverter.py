## Import libraries

import json
import os
import tqdm
from PIL import Image
from transformers import AutoProcessor
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D
from datasets import Dataset
from text_to_iob import text_to_iob

image_column_name = "image"
text_column_name = "tokens"
boxes_column_name = "boxes"
label_column_name = "labels"


funsd_labels = {
	'I-question':0,
	'B-question':1,
	'I-answer':2,
	'B-answer':3,
	'I-header':4,
	'B-header':5,
	'O':6
}


## PREPARING THE EXAMPLES FOR TRAINING
def prepare_examples(examples):
	images = examples[image_column_name]
	words = examples[text_column_name]
	boxes = examples[boxes_column_name]
	word_labels = examples[label_column_name]
	encoding = processor(images, words, boxes=boxes, word_labels=word_labels,
	               truncation=True, padding="max_length")
	return encoding

for data_type in ['train','test']:

	## STATICS
	IMAGES_DIR = f"dataset/{data_type}ing_data/images"

	## fetch names of all annotation files
	ANNOTATIONS_DIR = f"dataset/{data_type}ing_data/annotations"
	annotation_files = os.listdir(ANNOTATIONS_DIR)

	## globals to store the final data
	ALL_LABELS = []
	LAYOUT_ANNOTATIONS = []

	

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
			text = annotation['text']
			words = annotation['words']
			for word in words:
				BBOXES.append(word['box'])
				TOKENS.append(word['text'])
				label = annotation['label']
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
	batched =True,
	features = features,
	remove_columns = column_names
	)


	print(f"Saving {data_type} dataset to disk")

	final_data.save_to_disk(f'{data_type}.hf')