import pandas as pd 
import os
#os.environ["TRANSFORMERS_CACHE"] = '/cluster/scratch/niheil/.cache/huggingface/transformers/'
#os.environ["HF_HOME"] = '/cluster/scratch/niheil/.cache/huggingface'
from openai import OpenAI

df = pd.read_csv('essay_prompts_raw.csv', sep=';')

client = OpenAI(api_key=os.environ.get("TOGETHER_API_KEY"),
        base_url='https://api.together.xyz',
        )
model="meta-llama/Llama-2-70b-chat-hf"

for index, row in df.iterrows():
    messages=[
    {
      "role": "system",
      "content": "Your task is to write an essay (about 300-350 words) in response to a question.",
    },
    {
      "role": "user",
      "content": "Your task is to write an essay (about 300-350 words) in response to a question. The topic will be provided by the user.\nTopic:" + row.prompt_base,
    }]

    chat_completion = client.chat.completions.create(messages=messages, model="meta-llama/Llama-2-70b-chat-hf", max_tokens=512) 
    output = chat_completion.choices[0].message.content    
    print(index)
    print(output)
    df['essay_base'] = df['essay_base'].astype(str)
    df.at[index, 'essay_base'] = str(output)

for index, row in df.iterrows():
    messages=[
    {
      "role": "system",
      "content": "Your task is to write an essay (about 300-350 words) in response to a question.",
    },
    {
      "role": "user",
      "content": "Your task is to write an essay (about 300-350 words) in response to a question. The topic will be provided by the user.\nTopic:" + row.prompt_ESL,
    }]

    chat_completion = client.chat.completions.create(messages=messages, model="meta-llama/Llama-2-70b-chat-hf", max_tokens=512) 
    output = chat_completion.choices[0].message.content    
    print(index)
    print(output)
    df['essay_ESL'] = df['essay_ESL'].astype(str)
    df.at[index, 'essay_ESL'] = str(output)


for index, row in df.iterrows():
    messages=[
    {
      "role": "system",
      "content": "Your task is to write an essay (about 300-350 words) in response to a question.",
    },
    {
      "role": "user",
      "content": "Your task is to write an essay (about 300-350 words) in response to a question. The topic will be provided by the user.\nTopic:" + row.prompt_AAE,
    }]

    chat_completion = client.chat.completions.create(messages=messages, model="meta-llama/Llama-2-70b-chat-hf", max_tokens=512) 
    output = chat_completion.choices[0].message.content    
    print(index)
    print(output)
    df['essay_AAE'] = df['essay_AAE'].astype(str)
    df.at[index, 'essay_AAE'] = str(output)



df.to_csv('all_essays_llama.csv', sep=';')

