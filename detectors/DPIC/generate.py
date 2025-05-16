import argparse
import tqdm
import prompt
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def get_aux_llm(args):
    device_map = {'' : int(args.device.split(':')[1])}
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
            max_length=max_tokens,
            temperature=temperature,
            num_return_sequences=1,
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

def generate_text(model,tokenizer,guidingprompt):

    prompt = f"""
    You are given a **guiding prompt** that provides a clear direction for writing an article. \
    Your task is to use the guiding prompt to generate a well-structured, coherent, and informative article. \
    The article should fully address the key ideas suggested by the guiding prompt, providing thorough explanations, \
    examples, and insights where applicable, while remaining concise.

    The guiding prompt is as follows:
    {guidingprompt}

    Please generate an article based on the above guiding prompt. The article should:
    - Be written in a single, coherent paragraph without breaking into multiple sections.
    - Be well-structured, providing a logical flow from the beginning to the end.
    - Provide detailed explanations and relevant examples to support the ideas presented in the guiding prompt.
    - Avoid unnecessary jargon or complexity; aim for clarity and accessibility for a wide audience.
    - **The article should not exceed 800 words** in total. Please keep your writing concise while addressing the key points of the guiding prompt.
    
    Example of output format:
    Generate a single, well-structured paragraph summarizing and discussing the topic outlined in the guiding prompt.
    
    Finally, you only need to output the article you generated. \
    STOP immediately when the rewrite process is complete. \
    Do not include any extra information such as titles, comments, or additional notes at the beginning or end.\
    """

    template = base_prompt_template()
    query = template.format(query=prompt)

    response_content = get_response(model, tokenizer, query)
    return response_content

def run(args):

    tokenizer, model = get_aux_llm(args)
    df = prompt.run_get_prompt(model,tokenizer,args)

    guidingprompts = df['guiding_prompt'].values.tolist()
    generated_texts = []
    for guidingprompt in tqdm.tqdm(guidingprompts):
        generated_texts.append(generate_text(model,tokenizer,guidingprompt))
    df['dpic_text'] = generated_texts

    guidingprompts_2 = df['guiding_prompt_ai'].values.tolist()
    generated_texts_2 = []
    for guidingprompt in tqdm.tqdm(guidingprompts_2):
        generated_texts_2.append(generate_text(model,tokenizer,guidingprompt))
    df['dpic_text_ai'] = generated_texts_2

    print(df.head())
    print(df.shape)
    print(df.columns)

    return df





