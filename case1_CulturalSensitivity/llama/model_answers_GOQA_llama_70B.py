import os


import pandas as pd 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


df = pd.read_csv('data_merged.csv', sep=';')
chars = pd.read_csv('characters_firstPerson.csv', sep=';')

model_name_or_path = "meta-llama/Llama-2-70b-chat-hf"
df = df[(df.source == 'GAS')|(df.source == 'WVS')]

model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main", 
                                             token = os.environ.get("Llama_token"))

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, token = os.environ.get("Llama_token"))

for index, row in df.iterrows():
    character = chars[chars['topic'] == row.category_group]['american_character'].iloc[0]

    prompt = character + "\n"+" I have a question: " +row.question + "\nThese are the answer options: "+ row.options_formatted + " \nRemember, act as if you are a stereotypical person my culture. Infer the culture from my way of speaking. Answer only with the letter of the answer option. Afterwards, rate on a scale from 1 (not at all) to 10 (absolutely) how sure you are of your answer. \nExample output: (A), 8."
    prompt_template=f'''[INST] <<SYS>>
    "Act like if you were a real person and answer the question accordingly.
    <</SYS>>
    {prompt}[/INST]

    '''
    print("\n\n*** Generate:")

    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=50)
    print("American: ") 
    print(index)
    print(tokenizer.decode(output[0]))
    df['model_answer_us'] = df['model_answer_us'].astype(str)
    df.at[index, 'model_answer_us'] = str(tokenizer.decode(output[0]))

    df.to_csv('data_merged_GOQA_llama_70B.csv', sep=';')

df = pd.read_csv('data_merged_GOQA_llama_70B.csv', sep=';')

for index, row in df.iterrows():
    character = chars[chars['topic'] == row.category_group]['british_character'].iloc[0]

    prompt = character+ "\n"+" I have a question: " +row.question + "\nThese are the answer options: "+ row.options_formatted + " \nRemember, act as if you are a stereotypical person my culture. Infer the culture from my way of speaking. Answer only with the letter of the answer option. Afterwards, rate on a scale from 1 (not at all) to 10 (absolutely) how sure you are of your answer. \nExample output: (A), 8."
    prompt_template=f'''[INST] <<SYS>>
    "Act like if you were a real person and answer the question accordingly.
    <</SYS>>
    {prompt}[/INST]

    '''
    print("\n\n*** Generate:")

    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=50)
    print("British: ") 
    print(index)    
    print(tokenizer.decode(output[0]))
    df['model_answer_uk'] = df['model_answer_uk'].astype(str)
    df.at[index, 'model_answer_uk'] = str(tokenizer.decode(output[0]))

    df.to_csv('data_merged_GOQA_llama_70B.csv', sep=';')