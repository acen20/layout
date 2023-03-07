'''
Run inference on the model using the processor and model from above
'''
from transformers import AutoModelForTokenClassification
from transformers import LayoutLMv3ForTokenClassification
import torch
import os

label_list = ['I-question','I-answer','I-header','B-other']

id2label = {k: v for k,v in enumerate(label_list)}
label2id = {v: k for k,v in enumerate(label_list)}

model = LayoutLMv3ForTokenClassification.from_pretrained(
    "us_dl_model/checkpoint-500",
    id2label=id2label,
    label2id=label2id)

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
