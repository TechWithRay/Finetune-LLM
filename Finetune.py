from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import evaluate
import pandas as pd
import numpy as np


# Load the dataset
huggingface_dataset_name = "knkarthick/dialogsum"

dataset = load_dataset(huggingface_dataset_name)


# Load the model
model_name = "google/flan-t5-base"

original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Get the model and tokenizer parameters
def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    
    percentage = round((trainable_model_params / all_model_params) * 100)
    return f"trainable model parameters: {trainable_model_params} \nall model parameters: {all_model_params} \npercentage {percentage}"


# Get the tokenizer information


# Here we need to get the tokenizer information


# Zero Shot 


# One Shot


# Few Shot




## Full-Finetuning example

def tokenize_function(example):
    
    start_prompt = "Summarize the following conversation. \n\n"
    end_prompt = "\n\nSummary: "
    
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue]]
    
    example["input_ids"] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    
    example["labels"] = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    
    return example
    

# The dataset actually contains 3 diff splits: train, validation, test
# the tokenize_function code is handling all data across all splits in batches.

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_colums(["id", "topic", "dialogue", "summary"])


# To save some time in the lab.
tokenized_datasets = tokenized_datasets.filter(lambda example, index: index % 100 == 0, with_indices=True)

print(f"Shapes of the datasets: ")
print(f"Training: {tokenized_datasets["train"].shape}")
print(f"Validation: {tokenized_datasets["validation"].shape}")
print(f"Test: {tokenized_datasets["test"].shape}")

# Fine-tuning with the Preprocessed dataset
output_dir = f"./dialogue-summary-training-{str(int(time.time()))}"

training_args = TrainingArguments(

    output_dir=output_dir,
    learning_rate=1e-5,
    num_train_epochs=1,
    weight_decay=0.0.1,
    logging_steps=1,
    max_steps=1
)

trainer = Trainer(
    model=original_model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_dataset["validation"]
)

trainer.train()


# Quantitatively Evaluate the Model

dialogues = dataset["test"][:10]["dialogue"]
human_baseline_summaries = dataset["test"][:10]["summary"]

original_model_summaries = []
instruct_model_summaries = []


for _, dialogue in enumerate(dialogue):
    prompt = f"""
    
    Summarize the following conversation.
    
    {dialogue}
    
    Summary: """
    
    ## ?????????? what is the input_ids ???????????????
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    
    original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)
    original_model_summaries.append(original_model_text_output)
    
    instruct_model_outputs = instruct_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)
    instruct_model_summaries.append(instruct_model_text_output)
    
zipped_summaries = list(zip(human_baseline_summaries, original_model_summaries, instruct_model_summaries))

df = pd.DataFrame(zipped_summaries, columns= ["human_baseline_summaries", "original_model_summaries", "instruct_model_summaries"])

## qualitative checkout
original_model_results = rouge.compute(
    predictions=original_model_summaries,
    references=human_baseline_summaries[0:len(original_model_summaries)],
    use_aggregator=True,
    use_stemmer=True
)

instruct_model_results = rouge.compute(
    predictions=instruct_model_summaries,
    references=human_baseline_summaries[0:len(instruct_model_summaries)],
    use_aggregator=True,
    use_stemmer=True
)

print("ORIGINAL MODEL:")


print("INSTRUCT MODEL:")


## PEFT 

from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

peft_model = get_peft_model(original_model,
                            lora_config
)

print(print_number_of_trainable_model_parameters(peft_model))


## Lora finetuning
output_dir = f"./peft-dialogue-summary-training-{str(int(time.time()))}"

peft_training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3,  # Higher learning rate than full fine-tuning
    max_train_epochs=1,
    logging_steps=1,
    max_steps=1
)

peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_datasets["train"]
)

peft_trainer.train()



from peft import PeftModel, PeftConfig

peft_model_base = AutoModelForSeq2SeqLM.from_pretrained("google")





















