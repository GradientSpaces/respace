import logging
import wandb
import re
import json
import math
from pathlib import Path
import os
from transformers import TrainerCallback
import torch
import gc
import time
import pdb

from utils import remove_and_recreate_folder, get_model, safe_parse_scene
from test import run_test

def get_lora_sft_layers(model, process_index):
	model_modules = str(model.modules)
	pattern = r'\((\w+)\): Linear'
	linear_layer_names = re.findall(pattern, model_modules)
	names = []
	for name in linear_layer_names:
		names.append(name)
	target_modules = list(set(names))
	print(f"[ idx {process_index} ] lora layers for target_modules param:", target_modules)

class CustomTrainerCallback(TrainerCallback):
	def __init__(self, n_samples_snippet, trainer, dataset_train, dataset_val, dataset_test, sampling_engine, dvc, cli_args, accelerator=None, is_sft_training=False, log_every_n_steps=None, steps_per_epoch=None):
		self.trainer = trainer
		self.sampling_engine = sampling_engine
		self.cli_args = cli_args
		self.dvc = dvc
		self.accelerator = accelerator
		self.best_val_delta_pbl_loss = float("inf")
		self.n_samples_snippet = n_samples_snippet
		self.current_step_rewards = []

		self.dataset_train = dataset_train.select(range(self.n_samples_snippet))
		self.dataset_val = dataset_val.select(range(self.n_samples_snippet))
		self.dataset_test = dataset_test.select(range(self.n_samples_snippet))

		self.is_sft_training = is_sft_training
		self.log_every_n_steps = log_every_n_steps
		self.steps_per_epoch = steps_per_epoch

		self.all_prompts = json.load(open(os.getenv("PTH_ASSETS_METADATA_PROMPTS")))
		self.all_assets_metadata_simple_descs = json.load(open(os.getenv("PTH_ASSETS_METADATA_SIMPLE_DESCS")))
		
	def on_log(self, args, state, control, logs=None, **kwargs):
		if self.accelerator.is_main_process and logs is not None:
			current_epoch = int(math.ceil(state.epoch))

			if self.cli_args.use_wandb and logs and "loss" in logs:

				wandb.log({
					"train loss": logs["loss"],
					"train learning rate": logs["learning_rate"],
					"train grad norm": logs["grad_norm"],
					"epoch": current_epoch
				})
				if "reward" in logs:
					wandb.log({
						"train/grpo/reward": logs["reward"],
						"train/grpo/reward_std": logs["reward_std"],
						"epoch": current_epoch
					})
				if "rewards/chosen" in logs:
					wandb.log({
						"train/dpo/rewards_chosen": logs["rewards/chosen"],
						"train/dpo/rewards_rejected": logs["rewards/rejected"],
						"train/dpo/rewards_accuracies": logs["rewards/accuracies"],
						"train/dpo/rewards_margins": logs["rewards/margins"],
						"epoch": current_epoch
					})

	def on_step_end(self, args, state, control, **kwargs):
		# for GRPO / DPO
		if not self.is_sft_training and state.global_step > 0 and state.global_step % self.log_every_n_steps == 0:
			current_step = state.global_step
			virtual_epoch = current_step / self.steps_per_epoch
			
			print(f"\nRunning evaluation at step {current_step} (virtual epoch: {virtual_epoch:.2f})")	
			self._run_evaluation(state, virtual_epoch=virtual_epoch)

			if hasattr(self.trainer, "get_and_clear_step_rewards"):
				step_rewards = self.trainer.get_and_clear_step_rewards()
				
				if step_rewards and self.cli_args.use_wandb:
					wandb.log({
						"train/grpo/step_reward": step_rewards["mean"],
						"train/grpo/step_reward_std": step_rewards["std"],
						"step": state.global_step
					})
					print(f"Logging step rewards at step {state.global_step}: mean={step_rewards['mean']:.4f}, std={step_rewards['std']:.4f}")
		
		return control

	def on_evaluate(self, args, state, control, metrics=None, **kwargs):
		if self.is_sft_training:
			self.accelerator.wait_for_everyone()
			current_epoch = int(math.ceil(state.epoch))
			
			print(f"\nRunning evaluation at epoch {current_epoch}")
			self._run_evaluation(state, metrics=metrics)
		
		return control
	
	def _run_evaluation(self, state, virtual_epoch=None, metrics=None):
		# Your existing evaluation code from on_evaluate
		current_epoch = int(math.ceil(state.epoch)) if virtual_epoch is None else virtual_epoch
		checkpoint_dir_last = f"./ckpts/{self.cli_args.jid}/checkpoint-last"
		
		if self.accelerator.is_main_process:
			print(f"running custom eval at epoch {current_epoch})")
			save_model_and_config(checkpoint_dir_last, self.accelerator, self.trainer.model, self.trainer.processing_class)
			print("checkpoint-last saved!")

			if self.cli_args.use_wandb and metrics and "eval_loss" in metrics:
				eval_metrics = next((log for log in reversed(state.log_history) if any(k.startswith('eval_') for k in log.keys())), {})
				wandb.log({
					"eval loss": metrics["eval_loss"],
					"epoch": current_epoch
				})
				if "eval_reward" in eval_metrics:
					wandb.log({
						"eval/grpo/reward": eval_metrics["eval_reward"],
						"eval/grpo/reward_std": eval_metrics["eval_reward_std"],
						"epoch": current_epoch
					})
			
			# prep rendering folders if necessary
			if self.cli_args.do_renderings:
				for split in ["train", "val", "test"]:
					pth_viz_output = Path(f"{os.getenv('PTH_EVAL_VIZ_CACHE')}/run-test-subset-{split}")
					remove_and_recreate_folder(pth_viz_output)

		self.accelerator.wait_for_everyone()

		# **************************************************
		# run evaluation
		
		print(f"loading model from {checkpoint_dir_last} for evaluation...")

		eval_model, eval_tokenizer, max_seq_length = get_model(checkpoint_dir_last, self.cli_args.use_gpu, self.accelerator)
		if hasattr(eval_model, 'merge_and_unload'):
			print("calling merge_and_unload()...")
			eval_model_merged = eval_model.merge_and_unload()
			print("model merged!")
		else:
			print("using unmerged model...")
			eval_model_merged = eval_model
		
		aggregated_metrics_train = run_test(eval_model_merged, eval_tokenizer, self.accelerator, self.dvc, "train", self.cli_args.room_type, self.dataset_train, max_seq_length, self.sampling_engine, self.all_prompts, self.all_assets_metadata_simple_descs, self.cli_args.do_simple_descs, self.cli_args, n_best_of_n_llm=self.cli_args.n_best_of_n_llm, do_print=False, epoch=current_epoch)
		gc.collect()
		torch.cuda.empty_cache()
		time.sleep(3.0)
		
		aggregated_metrics_val = run_test(eval_model_merged, eval_tokenizer, self.accelerator, self.dvc, "val", self.cli_args.room_type, self.dataset_val, max_seq_length, self.sampling_engine, self.all_prompts, self.all_assets_metadata_simple_descs, self.cli_args.do_simple_descs, self.cli_args, n_best_of_n_llm=self.cli_args.n_best_of_n_llm, do_print=False, epoch=current_epoch)
		gc.collect()
		torch.cuda.empty_cache()
		time.sleep(3.0)

		aggregated_metrics_test = run_test(eval_model_merged, eval_tokenizer, self.accelerator, self.dvc, "test", self.cli_args.room_type, self.dataset_test, max_seq_length, self.sampling_engine, self.all_prompts, self.all_assets_metadata_simple_descs, self.cli_args.do_simple_descs, self.cli_args, n_best_of_n_llm=self.cli_args.n_best_of_n_llm, do_print=False, epoch=current_epoch)
		gc.collect()
		torch.cuda.empty_cache()
		time.sleep(3.0)

		val_delta_pbl_loss = aggregated_metrics_val["scene_delta_pbl_loss"]

		# save checkpoint-best
		if self.accelerator.is_main_process:
			if val_delta_pbl_loss < self.best_val_delta_pbl_loss:
				self.best_val_delta_pbl_loss = val_delta_pbl_loss
				checkpoint_dir = f"./ckpts/{self.cli_args.jid}/checkpoint-best"
				save_model_and_config(checkpoint_dir, self.accelerator, self.trainer.model, self.trainer.processing_class)
				print("checkpoint-best saved!")
				metadata = {
					"epoch": current_epoch,
					"best_val_delta_pbl_loss": val_delta_pbl_loss,
				}
				with open(os.path.join(checkpoint_dir, "best_val.json"), "w") as f:
					json.dump(metadata, f, indent=4)
			else:
				print("no improvement in val loss; model not saved")

		del eval_model_merged
		gc.collect()
		torch.cuda.empty_cache()
		time.sleep(3.0)

		print("\nEvaluation complete, resuming training...\n")
		self.accelerator.wait_for_everyone()

def save_model_and_config(checkpoint_dir, accelerator, model, tokenizer):
	os.makedirs(checkpoint_dir, exist_ok=True)
	unwrapped_model = accelerator.unwrap_model(model)
	unwrapped_model.save_pretrained(checkpoint_dir, save_embedding_layers=True)
	unwrapped_model.config.save_pretrained(checkpoint_dir)
	tokenizer.save_pretrained(checkpoint_dir)

def is_important_token(token):
	if token in [" ", ",", "[", "]", " [", "] ", "]}", "{", "}", "],", ' "', '"', '":', ""]:
		return False
	return True

def compute_custom_token_weights(inputs, processing_class):
	completion_ids = inputs["completion_ids"]
	completion_mask = inputs["completion_mask"]
	
	batch_size, seq_len = completion_mask.shape
	device = completion_mask.device

	token_weights = torch.zeros_like(completion_mask, dtype=torch.float32)
	text_per_token_lists = []
	
	for i in range(batch_size):
		text_per_token_list = processing_class.batch_decode(completion_ids[i].unsqueeze(1), skip_special_tokens=True)
		text_per_token_lists.append(text_per_token_list)
		full_completion = ''.join(text_per_token_list)

		# if it's not a valid json, skip and keep the default weights
		json_obj = safe_parse_scene(full_completion)
		if json_obj is None:
			token_weights[i, :] = 1.0
			continue

		idx_start_pos = text_per_token_list.index('pos')
		idx_start_rot = text_per_token_list.index('rot')
		idx_start_size = text_per_token_list.index('size')

		# print(idx_start_pos, idx_start_rot, idx_start_size)

		idxs_all_numerical = []

		for idx_token in range(idx_start_pos+2, idx_start_rot-2):
			if is_important_token(text_per_token_list[idx_token]):
				idxs_all_numerical.append(idx_token)

		for idx_token in range(idx_start_rot+2, idx_start_size-2):
			if is_important_token(text_per_token_list[idx_token]):
				idxs_all_numerical.append(idx_token)
		
		for idx_token in range(idx_start_size+2, len(text_per_token_list)-1):
			if is_important_token(text_per_token_list[idx_token]):
				idxs_all_numerical.append(idx_token)

		# print(idxs_all_numerical)
		# print("")

		# Apply weights to numerical tokens
		token_weights[i, idxs_all_numerical] = 1.0
	
	# # print each token and its weight on new line and split by delimiter for each item in batch
	# for k in range(batch_size):
	# 	for j in range(seq_len):
	# 		print(f"{j} \t {token_weights[k, j]} \t >>{text_per_token_lists[k][j]}<<")
	# 	print("\n=========================================================")
	# exit()
	
	# Apply completion mask to ensure we only weight actual tokens
	token_weights = token_weights * completion_mask
	
	return token_weights.to(device)