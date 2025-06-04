import pdb
import json
import re
import traceback
import math
import torch
from trl import GRPOTrainer
from src.dataset import WeightedRandomSampler
from torch.utils.data import DataLoader
import os
from trl import GRPOTrainer, GRPOConfig
import gc

from utils import safe_parse_scene, inherit_props_by_id, get_sft_model, init_wandb
from dataset import process_scene_sample, format_with_chat_template, create_full_scene_from_before_and_added
from eval import eval_scene_before_after_with_delta, compute_dss_score, compute_size_l2_dist
from train import CustomTrainerCallback, compute_custom_token_weights


class RewardFunctionWrapper:
	def __init__(self, trainer):
		self.trainer = trainer
		self.__name__ = "custom_reward_func"
	
	def __call__(self, completions, prompts, **kwargs):
		return self.trainer.calculate_reward(completions, prompts, **kwargs)

class GRPOSceneInstructionTrainer(GRPOTrainer):
	def __init__(self, *args, **kwargs):

		self.tok = kwargs.pop("tokenizer")
		self.max_seq_length = kwargs.pop("max_seq_length")

		self.sampling_engine = kwargs.pop("sampling_engine")
		self.n_best_of_n_assets = kwargs.pop("n_best_of_n_assets")
		self.do_augm = kwargs.pop("do_augm")
		self.do_simple_descs = kwargs.pop("do_simple_descs")
		self.do_full_sg_outputs = kwargs.pop("do_full_sg_outputs")
		
		self.do_token_masking = kwargs.pop("do_token_masking")
		self.do_neutral_rewards = kwargs.pop("do_neutral_rewards")
		
		self.all_prompts = json.load(open(os.getenv("PTH_ASSETS_METADATA_PROMPTS")))
		self.all_assets_metadata_simple_descs = json.load(open(os.getenv("PTH_ASSETS_METADATA_SIMPLE_DESCS")))
		
		self.neutral_indices = []
		self.reward_version = kwargs.pop("reward_version")

		# Create our wrapper 
		# original_reward_funcs = kwargs.get("reward_funcs")
		kwargs["reward_funcs"] = RewardFunctionWrapper(self)
		self.reward_step_buffer = []

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
	
	def calculate_reward(self, completions, prompts, **kwargs):
		rewards = [0.0] * len(completions)
		neutral_indices = []

		# print(self.original_data_mapping)
		
		# group completions by prompt
		prompt_to_completions = {}
		for i, prompt in enumerate(prompts):
			if prompt not in prompt_to_completions:
				prompt_to_completions[prompt] = []
			prompt_to_completions[prompt].append((i, completions[i]))
		
		for prompt, indexed_completions in prompt_to_completions.items():
			scene_before = json.loads(re.search(r'<scenegraph>(.*?)</scenegraph>', prompt, re.DOTALL).group(1))
			scene_before_with_assets = self.sampling_engine.sample_all_assets(scene_before, is_greedy_sampling=(True if self.n_best_of_n_assets == 1 else False))
			
			# orig_prompt = re.search(r'<add>(.*?)</add>', prompt, re.DOTALL).group(1)
			orig_prompt = self.original_data_mapping[prompt]["orig_prompt"]
			sg_output_add = json.loads(self.original_data_mapping[prompt]["sg_output_add"])

			for original_idx, completion in indexed_completions:
				unescaped_completion = completion.replace('\\"', '"')
				scene_after = safe_parse_scene(unescaped_completion)
				if scene_after is None:
					rewards[original_idx] = 0.0
				else:
					try:
						if scene_after.get("objects") is None:
							scene_after = create_full_scene_from_before_and_added(scene_before_with_assets, scene_after)

						# scene_after: sample last asset and inherit props from scene_before
						scene_after_with_assets = self.sampling_engine.sample_last_asset(scene_after, is_greedy_sampling=(True if self.n_best_of_n_assets == 1 else False))
						inherit_props_by_id(scene_before_with_assets, scene_after_with_assets)
						scene_after_with_assets["objects"][-1]["prompt"] = orig_prompt

						metrics = eval_scene_before_after_with_delta(scene_before_with_assets, scene_after_with_assets)

						# if self.reward_version == "v1":
						# 	rewards[original_idx] = 1.0 if metrics["delta_pbl_loss"] < 1e-5 else -1.0

						# V2: also consider PMS
						# if metrics["delta_pbl_loss"] < 1e-5 and metrics["txt_pms_score"] > 0.85:

						new_obj_desc = scene_after_with_assets["objects"][-1].get("desc")
						gt_obj_desc = sg_output_add.get("desc")
						txt_dss_score = compute_dss_score(new_obj_desc, gt_obj_desc, self.sampling_engine)
						
						new_obj_size = scene_after_with_assets["objects"][-1].get("size")
						gt_obj_size = sg_output_add.get("size")
						size_l2_dist = compute_size_l2_dist(new_obj_size, gt_obj_size)

						# print("delta pbl:", metrics["delta_pbl_loss"], "txt pms sampled score:", metrics["txt_pms_sampled_score"], "size l2 dist:", size_l2_dist, "txt dss score:", txt_dss_score)

						# V3: also consider size and DSS
						# if metrics["delta_pbl_loss"] < 1e-5 and metrics["txt_pms_score"] > 0.85 and size_l2_dist < 0.4 and txt_dss_score > 0.90:
						# 	raw_reward = 1.0
						# else:
						# 	raw_reward = -1.0

						# if self.reward_version == "v4":
						# 	if (metrics["delta_pbl_loss"] < 1e-5 and metrics["txt_pms_sampled_score"] > 0.85 and size_l2_dist < 0.2 and txt_dss_score > 0.90):
						# 		rewards[original_idx] = 1.0
						# 	else:
						# 		neutral_indices.append(original_idx)

						if self.reward_version == "v5":
							if (metrics["delta_pbl_loss"] < 1e-5 and metrics["txt_pms_sampled_score"] > 0.85 and size_l2_dist < 0.2 and txt_dss_score > 0.90):
								rewards[original_idx] = 1.0
							else:
								rewards[original_idx] = 0.2
								neutral_indices.append(original_idx)

					except Exception as e:
						print(f"error in eval_scene_before_after_with_delta: {e}")
						traceback.print_exc()
						rewards[original_idx] = 0.0

				# rewards[original_idx] = raw_reward
			
		if self.accelerator.is_main_process:
			self.reward_step_buffer.extend(rewards)
		
		self.neutral_indices = neutral_indices

		print("rewards at the end:", rewards)
		return rewards

	def _prepare_inputs(self, inputs):
		processed_inputs = []
		original_data_mapping = {}

		for sample in inputs:
			full_sample_instr, sample_sg_output_full, orig_prompt, _ = process_scene_sample(sample, self.tok, self.max_seq_length, self.all_prompts, self.all_assets_metadata_simple_descs, self.do_simple_descs, self.do_augm, self.do_full_sg_outputs)
			formatted_prompt = format_with_chat_template(self.processing_class, full_sample_instr)
			processed_sample = {"prompt": formatted_prompt}
			processed_inputs.append(processed_sample)

			original_data_mapping[formatted_prompt] = {
				"orig_prompt": orig_prompt,
				"sg_output_add": sample_sg_output_full,
			}

		self.original_data_mapping = original_data_mapping

		processed_result = super()._prepare_inputs(processed_inputs)

		# print("before: ", processed_result["advantages"])
		if self.do_neutral_rewards:
			for idx in self.neutral_indices:
				if idx < len(processed_result["advantages"]):
					processed_result["advantages"][idx] = 0.0
		# print("after: ", processed_result["advantages"])

		return processed_result

	# def print_top_tokens(self, logits, input_ids, completion_mask=None, top_k=10):
	# 	print("printing top-k tokens")
	# 	batch_size, seq_len, vocab_size = logits.shape
		
	# 	for batch_idx in range(batch_size):
	# 		print(f"\n----- Sample {batch_idx} -----")
			
	# 		# Decode the full completion for reference
	# 		if completion_mask is not None:
	# 			# Extract only the completion part
	# 			completion_len = completion_mask[batch_idx].sum().int().item()
	# 			completion = input_ids[batch_idx, -completion_len:]
	# 		else:
	# 			completion = input_ids[batch_idx]
				
	# 		# Get the actual tokens of the completion
	# 		completion_text = self.processing_class.decode(completion, skip_special_tokens=False)
	# 		print(f"Generated completion: {completion_text}")
			
	# 		# For each token position in the completion
	# 		for pos in range(completion.shape[0]):
	# 			# Skip padding tokens
	# 			if completion_mask is not None and not completion_mask[batch_idx, pos]:
	# 				continue
					
	# 			# Get the actual token at this position
	# 			token_id = completion[pos].item()
	# 			token_text = self.processing_class.decode([token_id], skip_special_tokens=False)
				
	# 			# Get the top-k token probabilities for this position
	# 			pos_logits = logits[batch_idx, -completion.shape[0] + pos]
	# 			top_values, top_indices = torch.topk(pos_logits, top_k)
	# 			top_probs = torch.softmax(top_values, dim=0)
				
	# 			print(f"\nPosition {pos}, Chosen token: '{token_text}' (ID: {token_id})")
	# 			print("Top-10 candidates:")
				
	# 			for i in range(top_k):
	# 				candidate_id = top_indices[i].item()
	# 				candidate_text = self.processing_class.decode([candidate_id], skip_special_tokens=False)
	# 				candidate_prob = top_probs[i].item()
	# 				candidate_logit = top_values[i].item()
					
	# 				# Mark the chosen token with an asterisk
	# 				marker = "*" if candidate_id == token_id else " "
	# 				print(f"{marker} {candidate_text:<10} (ID: {candidate_id:<6}) Prob: {candidate_prob:.4f} Logit: {candidate_logit:.4f}")

	# def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
	# 	# We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
	# 	logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
	# 	logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

	# 	# Get just the completion part of the input_ids
	# 	completion_ids = input_ids[:, -logits_to_keep:]
	# 	completion_mask = attention_mask[:, -logits_to_keep:]
		
	# 	# For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
	# 	logits = logits[:, -logits_to_keep:]
		
	# 	# Print top tokens for debugging (only for the first few batches to avoid flooding logs)
	# 	self.print_top_tokens(logits, completion_ids, completion_mask)
	# 	exit()

	def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

		# if not self.do_token_masking:
		
		return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
		
		# token_weights = compute_custom_token_weights(inputs, self.processing_class)

		# completion_mask = inputs["completion_mask"].clone()
		# token_weights = token_weights.to(completion_mask.device)
		# masked_completion_mask = completion_mask * token_weights

		# original_completion_mask = inputs["completion_mask"]
		# inputs["completion_mask"] = masked_completion_mask

		# loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

		# inputs["completion_mask"] = original_completion_mask

		# return loss

	# New method to get current step rewards for the callback
	def get_and_clear_step_rewards(self):
		if not self.reward_step_buffer:
			return None
		
		rewards = self.reward_step_buffer
		self.reward_step_buffer = []
		
		if len(rewards) > 0:
			mean_reward = sum(rewards) / len(rewards)
			std_reward = torch.tensor(rewards).std().item() if len(rewards) > 1 else 0.0
			return {
				"mean": mean_reward,
				"std": std_reward
			}
		return None

	
def run_grpo_training(model_id, tokenizer, accelerator, dataset_train, dataset_val, dataset_test, sampling_engine, dvc, args):

	accelerator.wait_for_everyone()
	gc.collect()
	torch.cuda.empty_cache()

	# **************************************************

	model, max_seq_length, lora_rank, lora_alpha = get_sft_model(model_id, args, accelerator)

	# **************************************************

	# fix issue with qwen that ignores padding left
	if args.do_grpo:
		tokenizer.padding_side = "left"
		if "qwen" in args.llm:
			model.config.use_cache = False

	# **************************************************

	learning_rate = args.grpo_learning_rate

	samples_per_step = args.dvc_batch_size * accelerator.num_processes * args.gas_steps
	steps_per_epoch = math.ceil(len(dataset_train) / samples_per_step)
	total_training_steps = steps_per_epoch * args.epochs
	
	log_every_n_steps = 5

	grpo_config = GRPOConfig(
		output_dir=f"./ckpts/{args.jid}",
		logging_dir=f"./logs/{args.jid}",

		num_train_epochs=args.epochs if not args.do_sanity_check else 1,
		max_steps=-1,
		
		per_device_train_batch_size=args.dvc_batch_size,
		per_device_eval_batch_size=args.dvc_batch_size,
		gradient_accumulation_steps=args.gas_steps,

		learning_rate=learning_rate,
		lr_scheduler_type="cosine",
		warmup_steps=5,
		
		# GRPO specific parameters
		beta=(0.05 if not args.grpo_do_neutral_rewards else args.grpo_beta_kl), # KL coefficient (from paper)
		num_generations=args.grpo_num_gen,

		# epsilon=0.05,
		# epsilon_high=0.2,
		
		# take same params as during evals
		temperature=args.grpo_temp,
		# temperature=0.9,
		# top_k=50,
		# top_p=0.95,

		max_prompt_length=max_seq_length,
		max_completion_length=128,
		
		save_strategy="no",
		logging_steps=1,
		logging_strategy="epoch",
		eval_strategy="epoch",
		greater_is_better=False,
		
		bf16=True,

		use_vllm=args.use_vllm,
		vllm_device=f"cuda:{accelerator.num_processes}" if args.use_vllm else "auto",
		vllm_max_model_len=max_seq_length,
		vllm_gpu_memory_utilization=0.8,
		
		# scale_rewards=False,
		# vllm_server_host=args.vllm_host,
		# vllm_server_port=int(args.vllm_port),
		# vllm_guided_decoding_regex = r'\{"desc": ".*?", "pos": \[-?\d+(\.\d+)?, -?\d+(\.\d+)?, -?\d+(\.\d+)?\], "rot": \[-?\d+(\.\d+)?, -?\d+(\.\d+)?, -?\d+(\.\d+)?, -?\d+(\.\d+)?\], "size": \[-?\d+(\.\d+)?, -?\d+(\.\d+)?, -?\d+(\.\d+)?\]\}',
		
		log_completions=False,
		remove_unused_columns=False,
		report_to=[],
		label_names=[],
	)

	# def wrapped_reward_func(prompts, completions, **kwargs):
	# 	return custom_reward_func(
	# 		completions=completions,
	# 		prompts=prompts,
	# 		sampling_engine=sampling_engine,
	# 		n_best_of_n_assets=1,
	# 		use_vlm=False,
	# 		**kwargs
	# 	)

	trainer = GRPOSceneInstructionTrainer(
		model=model,
		tokenizer=tokenizer,
		max_seq_length=max_seq_length,
		reward_funcs=None,
		train_dataset=dataset_train,
		eval_dataset=dataset_val.select(range(8)),
		args=grpo_config,
		processing_class=tokenizer,
		sampling_engine=sampling_engine,
		n_best_of_n_assets=1,
		do_augm=args.do_augm,
		do_simple_descs=args.do_simple_descs,
		do_full_sg_outputs=args.do_full_sg_outputs,
		do_token_masking=args.do_token_masking,
		do_neutral_rewards=args.grpo_do_neutral_rewards,
		reward_version=args.grpo_reward_version,
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
		print(f"GRPO: starting training")
		print(f"RUN ID (wandb): {args.run_id}")
		print(f"JOB ID (sherlock): {args.jid}")
		print(f"number of processes (accelerate): {accelerator.num_processes}")
		if args.use_lora:
			print(f"lora rank: {lora_rank}")
			print(f"lora alpha: {lora_alpha}")
			model.print_trainable_parameters()
		else:
			print("full number of params to train:", model.num_parameters())
		print(f"number of completions per prompt: {grpo_config.num_generations}")
		print(f"KL coefficient (beta): {grpo_config.beta}")
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
	 	print("GRPO TRAINING FINISHED!")