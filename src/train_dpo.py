import json
from trl import OnlineDPOTrainer, BasePairwiseJudge, OnlineDPOConfig
import torch
from torch.utils.data import DataLoader
import re
import math
import gc
import pdb
import os
import random
import traceback
from peft import get_peft_model
from datetime import datetime
import time
from vllm import LLM, SamplingParams
from accelerate import PartialState
from accelerate.utils import broadcast_object_list, gather, gather_object
import numpy as np

from utils import safe_parse_scene, inherit_props_by_id, get_sft_model, init_wandb, get_lora_config
from dataset import WeightedRandomSampler
from dataset import process_scene_sample, format_with_chat_template, create_full_scene_from_before_and_added
from eval import eval_scene_before_after_with_delta
from train import CustomTrainerCallback, compute_custom_token_weights

class OnlineDPOSceneTrainer(OnlineDPOTrainer):
	def __init__(self, *args, **kwargs):

		self.tokenizer = kwargs.pop("tokenizer")
		self.max_seq_length = kwargs.pop("max_seq_length")

		self.do_augm = kwargs.pop("do_augm")
		self.do_simple_descs = kwargs.pop("do_simple_descs")
		self.do_full_sg_outputs = kwargs.pop("do_full_sg_outputs")
		self.all_prompts = json.load(open(os.getenv("PTH_ASSETS_METADATA_PROMPTS")))
		self.all_assets_metadata_simple_descs = json.load(open(os.getenv("PTH_ASSETS_METADATA_SIMPLE_DESCS")))

		self.vllm_device = kwargs.pop("vllm_device")
		self.vllm_max_model_len = kwargs.pop("vllm_max_model_len")
		self.vllm_gpu_memory_utilization = kwargs.pop("vllm_gpu_memory_utilization")

		self.use_vllm = kwargs.pop("use_vllm")

		if self.use_vllm:
			gc.collect()
			torch.cuda.empty_cache()
			time.sleep(3.0)
		
			kwargs['args'].use_vllm = False
			super().__init__(*args, **kwargs)
			kwargs['args'].use_vllm = True

			if self.accelerator.is_main_process:
				print("overwrite existing vLLM setup with better props...")

				import contextlib
				import functools
				from unittest.mock import patch
		
				# Clean up GPU memory
				gc.collect()
				torch.cuda.empty_cache()
				time.sleep(3.0)
		
				device_type = PartialState().default_device.type
				device_module = getattr(torch, device_type)
		
				# Prepare all the patches from GRPOTrainer
				world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
				profiling_patch = patch(
					"vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", 
					return_value=None
				)
		
				# Create the NPU context manager if needed
				@contextlib.contextmanager
				def new_group_context():
					if not hasattr(torch, 'npu'):
						yield
						return
					
					new_group = torch.distributed.new_group
					try:
						torch.distributed.new_group = functools.partial(new_group, use_local_synchronization=True)
						torch.npu.mem_get_info = functools.partial(torch.npu.mem_get_info, device=self.vllm_device)
						yield
					finally:
						torch.distributed.new_group = new_group
				
				new_group_patch = new_group_context() if device_type == "npu" else contextlib.nullcontext()
				
				# Initialize vLLM with all the patches
				with world_size_patch, profiling_patch, new_group_patch:
					self.llm = LLM(
						model=self.model.name_or_path,
						device=self.vllm_device,
						gpu_memory_utilization=self.vllm_gpu_memory_utilization,
						dtype="auto",
						enable_prefix_caching=True,
						max_model_len=self.vllm_max_model_len,
					)
				
				# Create sampling parameters like in GRPO
				self.generation_config = SamplingParams(
					n=2,  # 2 generations per prompt
					max_tokens=self.args.max_new_tokens,
					temperature=self.args.temperature,
					top_k=50,
					top_p=0.95,
					detokenize=False,  # to avoid vllm decode (we don't need it)
				)

				print("vLLM setup done!")

			self.accelerator.wait_for_everyone()
		else:
			super().__init__(*args, **kwargs)
	
	# def get_train_dataloader(self):
	# 	if self.train_dataset is None:
	# 		return None
			
	# 	weights = torch.FloatTensor(self.train_dataset["sampling_weight"])
	# 	sampler = WeightedRandomSampler(
	# 		weights=weights,
	# 		num_samples=len(self.train_dataset),
	# 		replacement=True
	# 	)
	# 	return DataLoader(
	# 		self.train_dataset,
	# 		batch_size=self.args.per_device_train_batch_size,
	# 		sampler=sampler,
	# 		collate_fn=self.data_collator
	# 	)
	
	def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
		"""
		Intercept the inputs before they reach the original prediction_step to ensure
		they're in the correct format.
		"""
		# Check if inputs need preprocessing
		if isinstance(inputs, dict) and "prompt" not in inputs:
			inputs = self._preprocess_inputs_for_dpo(inputs)
			
			# Now convert prompts to the format needed by the model for evaluation
			tokenized = self.processing_class(
				inputs["prompt"], 
				return_tensors="pt", 
				padding=True, 
				truncation=True,
				max_length=self.max_length
			).to(model.device)
			
			# Create input format that's compatible with the model's forward method
			model_inputs = {
				"input_ids": tokenized["input_ids"],
				"attention_mask": tokenized["attention_mask"],
				# For evaluation, just use the input_ids as labels for a standard LM loss
				"labels": tokenized["input_ids"].clone()
			}
			
			# Now call the parent method with the properly formatted inputs
			return super().prediction_step(model, model_inputs, prediction_loss_only, ignore_keys)
		
		# If inputs are already correctly formatted, just call the parent method
		return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
	
	def training_step(self, model, inputs, num_items_in_batch=None):
		processed_inputs = self._preprocess_inputs_for_dpo(inputs)
		return super().training_step(model, processed_inputs, num_items_in_batch)
	
	def _preprocess_inputs_for_dpo(self, inputs):
		processed_inputs = {"prompt": []}

		num_samples = len(next(iter(inputs.values())))
		samples = []
		for i in range(num_samples):
			sample = {key: values[i] for key, values in inputs.items()}
			samples.append(sample)
		
		processed_inputs = {"prompt": []}
		for sample in samples:
			full_sample_instr, _, _, _ = process_scene_sample(sample, self.tokenizer, self.max_seq_length, self.all_prompts, self.all_assets_metadata_simple_descs, self.do_simple_descs, self.do_augm, self.do_full_sg_outputs)
			formatted_prompt = format_with_chat_template(self.processing_class, full_sample_instr)
			processed_inputs["prompt"].append(formatted_prompt)
		
		return processed_inputs
	
class SceneQualityJudge(BasePairwiseJudge):
	def __init__(self, sampling_engine, n_best_of_n_assets=1, use_vlm=False):
		self.sampling_engine = sampling_engine
		self.n_best_of_n_assets = n_best_of_n_assets
		self.use_vlm = use_vlm
		
	def judge(self, prompts, completion_pairs):
		
		# start_time = time.time()
		# readable_time = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
		# print(f"\njudge... (started: {readable_time})")

		results = []
		for prompt, (completion1, completion2) in zip(prompts, completion_pairs):

			# return random.choice([0, 1])
			# continue

			# Extract scene before from prompt
			scene_before = json.loads(re.search(r'<scenegraph>(.*?)</scenegraph>', prompt, re.DOTALL).group(1))
			scene_before_with_assets = self.sampling_engine.sample_all_assets(scene_before, is_greedy_sampling=(True if self.n_best_of_n_assets == 1 else False))

			scene_after1 = safe_parse_scene(completion1.replace('\\"', '"'))
			scene_after2 = safe_parse_scene(completion2.replace('\\"', '"'))
			
			# if one scene is None we automatically assign the other one as the better one
			if scene_after1 == None and scene_after2 == None:
				# print("JUDGE: both completions are None")
				results.append(random.choice([0, 1]))  # Random choice to avoid bias
				continue
			elif scene_after1 == None and scene_after2 != None:
				# print("JUDGE: completion 1 is None")
				results.append(1)
				continue
			elif scene_after1 != None and scene_after2 == None:
				# print("JUDGE: completion 2 is None")
				results.append(0)
				continue
			else:
				# process scene 1
				try:
					if scene_after1.get("objects") is None:
						scene_after1 = create_full_scene_from_before_and_added(scene_before_with_assets, scene_after1)
					scene_after1_with_assets = self.sampling_engine.sample_last_asset(scene_after1, is_greedy_sampling=(True if self.n_best_of_n_assets == 1 else False))
					inherit_props_by_id(scene_before_with_assets, scene_after1_with_assets)
					metrics1 = eval_scene_before_after_with_delta(scene_before_with_assets, scene_after1_with_assets)
					reward1 = metrics1["delta_pbl_loss"]
				except Exception as e:
					print(e)
					traceback.print_exc()
					reward1 = None

				# process scene 2
				try:
					if scene_after2.get("objects") is None:
						scene_after2 = create_full_scene_from_before_and_added(scene_before_with_assets, scene_after2)
					scene_after2_with_assets = self.sampling_engine.sample_last_asset(scene_after2, is_greedy_sampling=(True if self.n_best_of_n_assets == 1 else False))
					inherit_props_by_id(scene_before_with_assets, scene_after2_with_assets)
					metrics2 = eval_scene_before_after_with_delta(scene_before_with_assets, scene_after2_with_assets)
					reward2 = metrics2["delta_pbl_loss"]
				except Exception as e:
					print(e)
					traceback.print_exc()
					reward2 = None

				# print("JUDGE: reward1, reward2", reward1, reward2)

				if reward1 is None and reward2 is None or reward1 == reward2:
					results.append(random.choice([0, 1]))
				else:
					if reward1 == None and reward2 != None:
						results.append(1)
					elif reward1 != None and reward2 == None:
						results.append(0)
					else:
						if reward1 < reward2:
							results.append(0)
						else:
							results.append(1)

		print("results: ", results)
		# end_time = time.time()
		# elapsed_time = end_time - start_time
		# print(f"finished judge after {elapsed_time:.4f} s => {elapsed_time / len(results)} s / sample")
		
		return results

def run_dpo_training(model_id, tokenizer, accelerator, dataset_train, dataset_val, dataset_test, sampling_engine, dvc, args):
	
	accelerator.wait_for_everyone()
	gc.collect()
	torch.cuda.empty_cache()

	model, max_seq_length, lora_rank, lora_alpha = get_sft_model(model_id, args, accelerator)

	# peft_config = get_lora_config(lora_rank=256, lora_alpha=512)
	# model = get_peft_model(model, peft_config)

	# Fix issue with models that ignore padding left
	tokenizer.padding_side = "left"
	if "qwen" in args.llm:
		model.config.use_cache = False

	num_processes = accelerator.num_processes

	print("num_processes: ", num_processes)

	samples_per_step = args.dvc_batch_size * num_processes * args.gas_steps
	steps_per_epoch = math.ceil(len(dataset_train) / samples_per_step)
	total_training_steps = steps_per_epoch * args.epochs

	log_every_n_steps = 10
	# with 3 gpus and bs 8 and gas 4 we have 60 steps per epoch so we eval every 15% of the epoch

	dpo_config = OnlineDPOConfig(
		output_dir=f"./ckpts/{args.jid}",
		logging_dir=f"./logs/{args.jid}",
		
		num_train_epochs=args.epochs if not args.do_sanity_check else 2,
		max_steps=-1,
		
		per_device_train_batch_size=args.dvc_batch_size,
		per_device_eval_batch_size=args.dvc_batch_size,
		gradient_accumulation_steps=args.gas_steps,

		learning_rate=args.dpo_learning_rate,
		lr_scheduler_type="cosine",
		
		# DPO specific parameters
		beta=0.1,
		max_new_tokens=128,
		max_length=max_seq_length,
		temperature=0.7,
		
		save_strategy="no",
		logging_steps=steps_per_epoch,
		logging_strategy="epoch",
		eval_strategy="epoch",
		greater_is_better=False,
		
		bf16=True,
		
		use_vllm=args.use_vllm,

		ds3_gather_for_generation=True,
		
		remove_unused_columns=False,
		report_to=[],
		label_names=[],
	)

	# Create a judge for scene quality
	judge = SceneQualityJudge(
		sampling_engine=sampling_engine,
		n_best_of_n_assets=1,
		use_vlm=False
	)

	trainer = OnlineDPOSceneTrainer(
		model=model,
		tokenizer=tokenizer,
		max_seq_length=max_seq_length,
		judge=judge,
		args=dpo_config,
		processing_class=tokenizer,
		train_dataset=dataset_train,
		eval_dataset=dataset_val.select(range(4)),
		do_augm=args.do_augm,
		do_simple_descs=args.do_simple_descs,
		do_full_sg_outputs=args.do_full_sg_outputs,

		use_vllm=args.use_vllm,
		vllm_max_model_len=max_seq_length,

		# vllm_device="auto",
		# vllm_gpu_memory_utilization=0.15,

		vllm_device=f"cuda:{num_processes}" if args.use_vllm else "auto",
		vllm_gpu_memory_utilization=0.8,
	)

	trainer.model = accelerator.prepare(trainer.model)

	trainer.add_callback(CustomTrainerCallback(
		n_samples_snippet=min(500, len(dataset_test) if not args.do_sanity_check else 4),
		trainer=trainer,
		dataset_train=dataset_train,
		dataset_val=dataset_val,
		dataset_test=dataset_test,
		sampling_engine=sampling_engine,
		dvc=dvc,
		cli_args=args,
		accelerator=accelerator,
		is_sft_training=False,
		log_every_n_steps=log_every_n_steps,
		steps_per_epoch=steps_per_epoch,
	))

	init_wandb(args, accelerator)

	if accelerator.is_main_process:
		print("===============================================================================================")
		print(f"Online DPO: starting training")
		print(f"RUN ID (wandb): {args.run_id}")
		print(f"JOB ID (sherlock): {args.jid}")
		print(f"number of processes (accelerate): {num_processes}")
		if args.use_lora:
			print(f"lora rank: {lora_rank}")
			print(f"lora alpha: {lora_alpha}")
			model.print_trainable_parameters()
		else:
			print(f"full number of params to train: {model.num_parameters()}")
		print(f"KL coefficient (beta): {dpo_config.beta}")
		print(f"len of training dataset: {len(dataset_train)}")
		print(f"per device batch size: {args.dvc_batch_size}")
		print(f"samples_per_step: {samples_per_step}")
		print(f"steps_per_epoch: {steps_per_epoch}")
		print(f"log_every_n_steps: {log_every_n_steps}")
		print(f"total_training_steps: {total_training_steps}")
		print(f"using vLLM: {args.use_vllm}")
		print("===============================================================================================")
	
	accelerator.wait_for_everyone()
	trainer.train()
	
	if accelerator.is_main_process:
		print("ONLINE DPO TRAINING FINISHED!")