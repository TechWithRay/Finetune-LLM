#### Evaluation Scripts

index = 200
dialogue = dataset["test"][index]["dialogue"]
baseline_human_summary = dataset["test"][index]["summary"]


prompt = f"""
Summarize the following conversation.

{dialogue}

summary: """


input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# The model comparision
original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)


instruct_model_outputs = instruct_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)


peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

print(dash_line)
print(f"BASELINE HUMAN SUMMARY: \n {human_baseline_summary}")

print(dash_line)
print(f"ORIGINAL MODEL: \n {original_model_text_output}")

print(dash_line)
print(f"INSTRUCT MODEL: \n {instruct_model_text_output}")

print(dash_line)
print(f"PEFT MODEL: \n {peft_model_text_output}")



##################### Evaluate the Model Quantitatively (with ROUGE Metric) ###########################

# perform inference for sample of the test dataset

dialogue = dataset["test"][0:10]["dialogue"]
human_baseline_summaries = dataset["test"][0:10]["summary"]


original_model_summaries = []
instruct_model_summaries = []
peft_model_summaries = []


for idx, dialogue in enumerate(dialogues):
    prompt = f"""
    Summarize the following conversation.
    
    {dialogue}
    
    Summary: """

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    
    human_baseline_text_output = human_baseline_summaries[idx]
    
    original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)
    
    instruct_model_outputs = instruct_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)
    
    peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)
    
    
    # add to the list
    original_model_summaries.append(original_model_text_output)
    instruct_model_summaries.append(instruct_model_text_output)
    peft_model_summaries.append(peft_model_text_output)


zipped_summaries = list(zip(original_model_summaries, instruct_model_summaries, peft_model_summaries))

df = pd.DataFrame(zipped_summaries, colums=["human_baseline_summaries" "original_model_summaries", "instruct_model_summaries", "peft_model_summaries"])


########################## Calculate the ROUGE metrics #############################

human_baseline_rouge = human_baseline_summaries[0:len(original_model_summaries)]


original_model_results = rouge.compute(
    predictions=original_model_summaries,
    references=human_baseline_rouge,
    use_aggregator=True,
    use_stemmer=True
)

instruct_model_results = rouge.compute(
    predictions=instruct_model_summaries,
    references=human_baseline_rouge,
    use_aggregator=True,
    use_stemmer=True
)

peft_model_results = rouge.compute(
    predictions=peft_model_summaries,
    references=human_baseline_summaries[0:len(peft_model_summaries)],
    use_aggregator=True,
    use_stemmer=True
)


print("ORIGINAL MODEL: ")
print(original_model_results)

print("INSTRUCT MODEL:")
print(instruct_model_results)

print("PEFT MODEL")
print(peft_model_results)








