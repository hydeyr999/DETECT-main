import pandas as pd
import ast
import tqdm
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from data import *

def get_aux_llm(args):
    device_map = {"": 1}
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16"
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        use_cache=False,
        device_map=device_map
    )


    return tokenizer, model

def get_response(model,tokenizer,prompt,temperature = 0.2,max_tokens = 3000):
    max_embedding_length =int(max_tokens / 2)
    encoding = tokenizer(prompt, return_tensors="pt",max_length=max_embedding_length,truncation=True,padding='max_length')
    input_ids = encoding['input_ids'].to(model.device)
    attention_mask = encoding['attention_mask'].to(model.device)
    # 生成文本
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_tokens,  # 输出的最大长度
            temperature=temperature,  # 控制输出的创意程度
            num_return_sequences=1,  # 生成一个结果
            no_repeat_ngram_size=2,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    content = generated_text.split("<|assistant|>")[-1].strip()

    return content

def base_prompt_template() -> str:
    template = """<|system|>
    You are a large language model trained by microsoft. Follow the user's instructions carefully. Respond using markdown.
    <|user|>
    {query}
    <|assistant|>
    """
    return template

def get_prompt(model,tokenizer,text):
    prompt = f"""
                You are given a piece of text, and your task is to generate a **guiding prompt** that summarizes the key ideas of the article. This prompt should be informative and specific, \
                offering a clear direction for further analysis or content generation based on the article's content. \
                
                The input text is as follows: \
                {text} 
                
                Please generate a guiding prompt for this article. The prompt should:\
                - Be concise but informative, capturing the essence of the article's main points.\
                - Offer a direction for further analysis or thought based on the content of the article.\
                - Help users understand the key themes of the article and how they might approach generating more content or insights related to it.\
                
                Example of output format:\
                "Based on the given article, create a prompt that asks about [main theme or idea], focusing on [specific aspect or question]."\
                
                Finally, you only need to output your guiding prompt. \
                STOP immediately when the rewrite process is complete. \
                Do not include any extra information such as titles, comments, or additional notes at the beginning or end.\
                Just focus on providing the guiding prompt only.
            """

    template = base_prompt_template()
    query = template.format(query=prompt)

    response_content = get_response(model, tokenizer, query)
    return response_content

def run_get_prompt(model,tokenizer,args):
    df = get_df(args)

    ai_texts = []
    for _, context in tqdm.tqdm(df.iterrows(), total=len(df)):
        context = context.tolist()[0]
        ai_texts.append(get_prompt(model, tokenizer, context))
    df['guiding_prompt'] = ai_texts

    ai_texts_2 = []
    for _, context in tqdm.tqdm(df.iterrows(), total=len(df)):
        context = context.tolist()[1]
        ai_texts_2.append(get_prompt(model, tokenizer, context))
    df['guiding_prompt_ai'] = ai_texts_2

    print(df.head())
    print(df.shape)
    print(df.columns)

    return df
