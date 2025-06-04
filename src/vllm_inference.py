import torch
from vllm import LLM, SamplingParams
import os
import time
from datetime import datetime
import gc

class VLLMWrapper:
    def __init__(self, model_id, tokenizer, gpu_memory_utilization=0.8, max_model_len=None):
        print(f"Initializing vLLM with model {model_id}")
        try:
            # # For multi-GPU setups (A100s)
            # if tensor_parallel_size > 1:
            #     self.llm = LLM(
            #         model=model_id,
            #         tensor_parallel_size=tensor_parallel_size,
            #         gpu_memory_utilization=gpu_memory_utilization,
            #         max_model_len=max_model_len
            #     )
            # else:
            # For single GPU (4090)
            self.llm = LLM(
                model=model_id,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len
            )
            self.tokenizer = tokenizer
            self.initialized = True
            print("vLLM initialized successfully")
        except Exception as e:
            print(f"Error initializing vLLM: {e}")
            self.initialized = False
    
    def generate(self, input_ids, attention_mask, max_new_tokens, temperature=0.7, top_p=0.95, top_k=50, do_sample=True):
        if not self.initialized:
            raise RuntimeError("vLLM not properly initialized")
        
        # Convert input_ids to prompts
        # Note: vLLM doesn't work directly with input_ids, so we need to get the tokenizer output
        # You might need to adjust this part based on how you want to handle the conversion
        prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        
        # Setup sampling parameters
        sampling_params = SamplingParams(
            n=1,  # Number of completions per prompt
            max_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            top_p=top_p if do_sample else 1.0,
            top_k=top_k if do_sample else -1,
        )
        
        # Generate completions
        start_time = time.time()
        outputs = self.llm.generate(prompts, sampling_params)
        end_time = time.time()
        
        # Process outputs
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            results.append(generated_text)
        
        print(f"vLLM generation completed in {end_time - start_time:.2f}s")
        return results
    
    def __del__(self):
        """Clean up resources"""
        if hasattr(self, 'llm') and self.llm is not None:
            del self.llm
            gc.collect()
            torch.cuda.empty_cache()
            print("vLLM resources cleaned up")