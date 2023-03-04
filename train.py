from docai.training import generate_layoutlm_compute_eval_metric_fn
from datasets import load_from_disk
# Load the microsoft/layoutlmv3-base processor from the Hugging Face hub
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)


train_dataset = load_from_disk('train.hf')
eval_dataset = load_from_disk('test.hf')


label_list = ['I-question','I-answer','I-header','B-other']

id2label = {k: v for k,v in enumerate(label_list)}
label2id = {v: k for k,v in enumerate(label_list)}

# Use this utility from the docai SDK to create a function that can
# be used to calculate the evaluation metrics while training
compute_eval_metrics = generate_layoutlm_compute_eval_metric_fn(
    ner_labels=label_list,
    metric_name="seqeval",
    return_entity_level_metrics=False
)


'''
Define our model, as well as the TrainingArguments which includes all the 
hyperparameters related to training.
'''

from transformers import TrainingArguments, Trainer
from transformers import LayoutLMv3ForTokenClassification
from transformers.data.data_collator import default_data_collator

MODEL_NAME = 'us_dl_model'

model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    id2label=id2label,
    label2id=label2id)

training_args = TrainingArguments(
    output_dir=MODEL_NAME,
    max_steps=1000,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=1e-5,
    evaluation_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1")

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor,
    data_collator=default_data_collator,
    compute_metrics=compute_eval_metrics,
)

'''
Train the model!
'''
trainer.train()
