import json
import time
from openai import OpenAI
import concurrent.futures
import threading
from tqdm import tqdm
import numpy as np

class ThreadSafeDict:
    def __init__(self):
        self.data = {}
        self.lock = threading.Lock()

    def __setitem__(self, key, value):
        with self.lock:
            self.data[key] = value

    def __getitem__(self, key):
        with self.lock:
            return self.data[key]

    def get(self, key, default=None):
        with self.lock:
            return self.data.get(key, default)

def thread_safe_file_write(file_path, data):
    with threading.Lock():
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

def get_gpt4_response(desc, openai_client):
    answer = None
    
    while answer is None:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user", 
                    "content": f"""
                    The following description describes a furniture or object. Please extract the main subject such that your final output is only one word. If possible, use one of the labels from the NYU40 labels from ScanNetV2. Otherwise, just use your best judgment. Only return the label in lowercase and nothing else.
                    Text: '{desc}'
                    """
                }],
                temperature=0.3
            )
            
            answer = response.choices[0].message.content.strip()
            
        except Exception as exc:
            print(f"Error in GPT-4 request: {exc}")
            print("Retrying...")
            
        if answer is None:
            t_sleep = np.random.randint(5, 9)
            print(f"sleeping {t_sleep}s...")
            time.sleep(t_sleep)
    
    return answer

def process_entry(args):
    idx, (key, entry), lookup_table, openai_client = args
    try:
        long_desc = entry.get('summary')
        if not long_desc:
            return
        
        short_desc = get_gpt4_response(long_desc, openai_client)
        if short_desc:
            lookup_table[long_desc] = short_desc
            #print(f"Processed entry {idx}")
            #print(f"Long: {long_desc[:50]}...")
            #print(f"Short: {short_desc}")
            #print("-" * 50)
            
    except Exception as e:
        print(f"Error processing entry {idx}: {str(e)}")

def create_lookup_table(input_file, output_file, num_threads=16, test_mode=True):
    openai_client = OpenAI(api_key="sk-proj-n0LSwBDfjFbxT6Xx4KkwT3BlbkFJM1qUnlM8RaaXcO5A6TCz")
    
    # Load data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Create thread-safe lookup dictionary
    lookup_table = ThreadSafeDict()
    
    # Prepare items to process
    items = list(enumerate(data.items()))

    if test_mode:
        # In test mode, process only num_threads items
        items = items[:num_threads]
    
    total_items = len(items)
    print(f"Processing {total_items} items with {num_threads} threads")

    # Process items using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        args_list = [(i, item, lookup_table, openai_client) for i, item in items]
        list(tqdm(executor.map(process_entry, args_list), total=total_items))

    # Save results
    thread_safe_file_write(output_file, lookup_table.data)
    print(f"\nLookup table saved to {output_file}")
    print(f"Processed {len(lookup_table.data)} entries successfully")

if __name__ == "__main__":
    input_file = "/Volumes/apollo11/data/3D-FUTURE-assets/model_info_martin.json"
    output_file = "/Volumes/apollo11/data/3D-FUTURE-assets/model_info_martin_simple_descs.json"
    
    # # Test run with different thread counts
    # thread_counts = [ 16 ]
    # for threads in thread_counts:
    #     print(f"\nTesting with {threads} threads...")
    #     create_lookup_table(input_file, output_file, num_threads=threads, test_mode=True)
    #     time.sleep(2)  # Brief pause between tests
    
    create_lookup_table(input_file, output_file, num_threads=16, test_mode=False)