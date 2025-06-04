import os
from openai import OpenAI
import json
import time
import numpy as np
from dotenv import load_dotenv
import concurrent.futures
from typing import List, Dict
from tqdm import tqdm

N_PROMPTS_PER_OBJ = 10
PROMPT_MOD_PREFIX = "The list below contains a sentence referencing to a single piece of furniture. Your task it to create a list of 10 short descriptions that vary in length. Each description refers to the subject with a maximum of 3-4 additional descriptive words that reference the colour, style, shape, etc. All your sentences should be in 'noun phrase'. You MUST include a variety of lengths in your descriptions, ensuring a few samples are very short (1-2 words max) and others are longer (4-5 words). Have at least one sample with only one word, except if you need to be more specific for the subject, e.g. use 'Coffee Table', not just 'Table', if present. Use mostly basic properties such as colour or material, but also include a few creative and diverse versions to increase robustness in our ML training dataset.\n\nThe sentence is:\n\n"
PROMPT_MOD_POSTFIX = f"\nJust output a plain list and nothing else. You have only one list of {N_PROMPTS_PER_OBJ} descriptions. You MUST always point to the referenced object above and not hallucinate other furniture or be overly generic by using 'furniture' or 'piece'. Every list contains the descriptions in increasing word length. Just output the final JSON object as a plain string without any key. Never use markdown or ```json."

def get_list_of_obj_descs(all_descs: List[str]) -> str:
    return "".join([f"- {desc}\n" for desc in all_descs])

def do_gpt4_request(openai_client: OpenAI, full_prompt: str, temp: float = 1.0):
    resp = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    { 
                        "type": "text", "text": full_prompt 
                    },
                ],
            }
        ],
        temperature=temp
    )
    return resp

def get_gpt4_resp(openai_client: OpenAI, full_prompt: str):
    answer_extr = None
    resp = None
    tmp = 1.0
    while answer_extr is None:
        try:
            response = do_gpt4_request(openai_client, full_prompt, tmp)
            resp = json.loads(response.json())
            answer = resp.get("choices")[0].get("message").get("content")
            answer_extr_tmp = json.loads(answer)
            # print(answer_extr_tmp)
            if isinstance(answer_extr_tmp, list) and len(answer_extr_tmp) == N_PROMPTS_PER_OBJ and isinstance(answer_extr_tmp[0], str) and len(answer_extr_tmp[0]) > 1:
                answer_extr = answer_extr_tmp
            else:
                print("didn't match specs!")
        except Exception as exc:
            # if resp is not None:
                # print(resp)
            print(exc)
            print("exception. trying again...")
        if answer_extr is None:
            t_sleep = np.random.randint(5, 9)
            print(f"couldn't extract answer. trying again after {t_sleep}s...")
            time.sleep(t_sleep)
    return answer_extr

def process_single_item(args: tuple) -> tuple:
    key, desc, openai_client = args
    full_prompt = f"{PROMPT_MOD_PREFIX}{get_list_of_obj_descs([ desc ])}{PROMPT_MOD_POSTFIX}"
    prompts = get_gpt4_resp(openai_client, full_prompt)
    return key, [prompt.lower() for prompt in prompts]

def build_prompts_parallel(metadata_all: Dict, openai_client: OpenAI, max_workers: int = 32) -> Dict:
    metadata_prompts = {}
    total_items = len(metadata_all)
    
    # Prepare arguments for parallel processing
    process_args = [
        (key, metadata_all[key]["summary"], openai_client) 
        for key in metadata_all.keys()
    ]
    
    # Process items in parallel with tqdm progress bar
    with tqdm(total=total_items, desc="Processing items") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_key = {
                executor.submit(process_single_item, args): args[0] 
                for args in process_args
            }
            
            for future in concurrent.futures.as_completed(future_to_key):
                key, prompts = future.result()
                metadata_prompts[key] = prompts
                pbar.update(1)
    
    return metadata_prompts

if __name__ == "__main__":
    load_dotenv(".env.local")
    
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    metadata_all = json.load(open(os.getenv("PTH_ASSETS_METADATA")))

    metadata_prompts = build_prompts_parallel(metadata_all, openai_client)
    
    with open(os.getenv("PTH_ASSETS_METADATA_PROMPTS"), "w") as f:
        json.dump(metadata_prompts, f)