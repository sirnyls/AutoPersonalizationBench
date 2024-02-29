import openai
import os
import pandas as pd
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def write_essay(input, model="gpt-4-1106-preview"):
    instruction = "Your task is to write an essay (about 300-350 words) in response to a question. The topic will be provided by the user."    
    response = client.chat.completions.create(
    model=model,
    messages=[
        {
        "role": "system",
        "content": instruction
        },
        {
        "role": "user",
        "content": "Your task is to write an essay (about 300-350 words) in response to a question. The topic will be provided by the user. \nTopic:" + input
        }
    ],
    max_tokens=500,
    temperature=1
    )
    #print(response.choices[0].message.content)
    # Extracting the similarity score from the response
    result = response.choices[0].message.content
    print(result)
    return result

def apply_write_essay(row, prompt):
    try:
        return write_essay(row[prompt])
    except Exception as e:
        print(f"Error processing row: {e}")
        return None
df = pd.read_csv('../essay_prompts_raw.csv')

df['essay_base'] = df.apply(apply_write_essay('prompt_base'), axis=1)
df['essay_AAE'] = df.apply(apply_write_essay('prompt_AAE'), axis=1)
df['essay_ESL'] = df.apply(apply_write_essay('prompt_ESL'), axis=1)

df.to_csv('../all_essays_gpt.csv')
