import torch
from trl import SFTTrainer, SFTConfig
from torch.utils.data import DataLoader
import math
from peft import get_peft_model
import json
import os
import numpy as np
import random

from utils import init_wandb, get_lora_config
from train import CustomTrainerCallback
from src.dataset import SFTSceneDataCollator, process_scene_sample

# class SFTSceneInstructionTrainer(SFTTrainer):
# 	def __init__(self, *args, **kwargs):
# 		super().__init__(*args, **kwargs)

# 	def get_train_dataloader(self):
# 		weights = torch.FloatTensor(self.train_dataset["sampling_weight"])
# 		sampler = WeightedRandomSampler(
# 			weights=weights,
# 			num_samples=len(self.train_dataset),
# 			replacement=True
# 		)
# 		return DataLoader(
# 			self.train_dataset,
# 			batch_size=self.args.per_device_train_batch_size,
# 			sampler=sampler,
# 			collate_fn=self.data_collator
# 		)

# 	# for eval/test, just use regular sampling (one instruction per scene)
# 	def get_eval_dataloader(self, eval_dataset=None):
# 		return super().get_eval_dataloader(eval_dataset)

def verify_augmentation_working(dataset_train, tokenizer, max_seq_length, args):
	"""Rigorous verification that augmentation produces different geometries for the SAME object.

	The key insight: we need to ensure create_instruction_from_scene() picks the same object
	in both passes, then compare how that identical object gets augmented differently.
	"""

	print("\n" + "="*80)
	print("VERIFYING DATA AUGMENTATION IS WORKING")
	print("="*80 + "\n")

	all_prompts = json.load(open(os.getenv("PTH_ASSETS_METADATA_PROMPTS")))
	all_assets_metadata_simple_descs = json.load(open(os.getenv("PTH_ASSETS_METADATA_SIMPLE_DESCS")))

	# Pick 5 random samples to verify
	num_samples_to_check = 5
	sample_indices = np.random.choice(len(dataset_train), num_samples_to_check, replace=False)

	samples_pass1 = []
	samples_pass2 = []

	# Pass 1: Process samples
	print("Pass 1: Processing samples with FIXED random state for object selection...")
	for idx in sample_indices:
		orig_sample = dataset_train[int(idx)]

		# CRITICAL: Set seeds BEFORE process_scene_sample to control object selection
		# This ensures create_instruction_from_scene picks the same object in both passes
		np.random.seed(1234 + int(idx))  # Use idx-specific seed for diversity
		random.seed(1234 + int(idx))
		torch.manual_seed(1234 + int(idx))

		full_instr, completion, prompt, sample = process_scene_sample(
			orig_sample, tokenizer, max_seq_length,
			all_prompts, all_assets_metadata_simple_descs,
			args.do_simple_descs, True, args.do_full_sg_outputs
		)

		samples_pass1.append({
			'idx': idx,
			'sg_output_add': completion,
			# 'sg_output_add': sample['sg_output_add'],
			# 'sg_input': sample['sg_input'],
			# 'desc': json.loads(sample['sg_output_add'])['desc']
		})

	# Pass 2: Process same samples again with SAME seeds for object selection
	print("Pass 2: Processing same samples again with SAME object selection seeds...")
	for idx in sample_indices:
		orig_sample = dataset_train[int(idx)]

		# Use SAME seed as Pass 1 to ensure same object is selected
		np.random.seed(1234 + int(idx))
		random.seed(1234 + int(idx))
		torch.manual_seed(1234 + int(idx))

		full_instr, completion, prompt, sample = process_scene_sample(
			orig_sample, tokenizer, max_seq_length,
			all_prompts, all_assets_metadata_simple_descs,
			args.do_simple_descs, False, args.do_full_sg_outputs
		)

		samples_pass2.append({
			'idx': idx,
			'sg_output_add': completion,
			#'sg_output_add': sample['sg_output_add'],
			#'sg_input': sample['sg_input'],
			#'desc': json.loads(sample['sg_output_add'])['desc']
		})

	# Compare the two passes
	print("\n" + "-"*80)
	print("COMPARISON RESULTS:")
	print("-"*80 + "\n")

	different_count = 0
	same_object_count = 0

	for i, (s1, s2) in enumerate(zip(samples_pass1, samples_pass2)):
		# Extract object data from sg_output_add
		obj1 = json.loads(s1['sg_output_add'])
		obj2 = json.loads(s2['sg_output_add'])

		# CRITICAL CHECK: Verify we're comparing the SAME object
		same_object = (obj1['desc'] == obj2['desc'])
		if same_object:
			same_object_count += 1

		# Compare geometries
		pos_diff = np.linalg.norm(np.array(obj1['pos']) - np.array(obj2['pos']))
		size_diff = np.linalg.norm(np.array(obj1['size']) - np.array(obj2['size']))

		# Compare rotations (quaternions)
		rot1 = np.array(obj1['rot'])
		rot2 = np.array(obj2['rot'])
		rot_diff = np.linalg.norm(rot1 - rot2)

		is_different = (pos_diff > 0.001 or size_diff > 0.001 or rot_diff > 0.001)

		if is_different:
			different_count += 1

		print(f"Sample {i} (idx={s1['idx']}):")
		print(f"  Object match: {obj1['desc'][:50]} {'==' if same_object else '!='} {obj2['desc'][:50]}")
		if same_object:
			print(f"  Geometry: {'DIFFERENT' if is_different else 'IDENTICAL'}")
			if is_different:
				print(f"    Position diff: {pos_diff:.4f}")
				print(f"    Size diff: {size_diff:.4f}")
				print(f"    Rotation diff: {rot_diff:.4f}")
		else:
			print(f"  ✗ WARNING: Different objects selected (seed control failed!)")
		print()

	print("-"*80)
	print(f"Object selection consistency: {same_object_count}/{num_samples_to_check} samples had same object")

	if same_object_count < num_samples_to_check:
		print("✗ CRITICAL: Seed control failed - not comparing same objects across passes!")
		print("  The verification is invalid.")
	elif args.do_augm:
		if different_count == num_samples_to_check:
			print(f"✓ AUGMENTATION WORKING: All {num_samples_to_check} samples produced different geometries")
			print("  The SAME objects were augmented differently across passes.")
		else:
			print(f"✗ AUGMENTATION ISSUE: Only {different_count}/{num_samples_to_check} samples were different")
			print("  Augmentation might not be applying geometric noise consistently.")
	else:
		if different_count == 0:
			print(f"✓ NO AUGMENTATION: All {num_samples_to_check} samples were identical (as expected)")
		else:
			print(f"✗ UNEXPECTED: {different_count}/{num_samples_to_check} samples were different despite no augmentation")
			print("  This indicates unwanted randomness in the pipeline.")

	print("-"*80 + "\n")

	exit()


def run_sft_training(model, model_id, max_seq_length, tokenizer, accelerator, dataset_train, dataset_val, dataset_test, sampling_engine, dvc, args):

	# VERIFICATION: Check if augmentation is working (only on main process, only in sanity check mode)
	# if accelerator.is_main_process and args.do_sanity_check:
	# 	verify_augmentation_working(dataset_train, tokenizer, max_seq_length, args)
	# exit()

	# LLPlace Paper:
	# LoRA alpha value to 32, 
	# the LoRA r value to 8, 
	# and the LoRA dropout rate to 0.05. 
	# Additionally, we set the learning rate to 1e-4 and use a cosine scheduler for optimization, training for 20 epochs.
	
	# learning_rate = 1e-4 #2e-5
	learning_rate= 5e-5

	per_device_batch_size = args.dvc_batch_size
	gas_steps = args.gas_steps
	learning_rate *= accelerator.num_processes

	samples_per_step = per_device_batch_size * accelerator.num_processes * gas_steps
	steps_per_epoch = math.ceil(len(dataset_train) / samples_per_step)
	total_training_steps = steps_per_epoch * args.epochs

	if "llama" in model_id:
		response_template = "<|start_header_id|>user<|end_header_id|>\n\n"
	else:
		response_template = "<|im_start|>assistant\n"

	data_collator = SFTSceneDataCollator(
		do_augm=args.do_augm,
		response_template=response_template,
		tokenizer=tokenizer,
		padding_free=True,
		max_seq_length=max_seq_length,
		do_simple_descs=args.do_simple_descs,
		do_full_sg_outputs=args.do_full_sg_outputs,
	)

	lora_rank = args.lora_rank if args.lora_rank is not None else 16
	lora_alpha = args.lora_alpha if args.lora_alpha is not None else lora_rank * 2

	if args.use_lora:
		peft_config = get_lora_config(lora_rank, lora_alpha)
		model = get_peft_model(model, peft_config)
	else:
		peft_config = None

	sft_config = SFTConfig(
		# neftune_noise_alpha=5,
		output_dir=f"./ckpts/{args.jid}",
		logging_dir=f"./logs/{args.jid}",

		max_seq_length=max_seq_length,

		packing=False,
		bf16=True,

		lr_scheduler_type="cosine",

		learning_rate=learning_rate,
		warmup_steps=0,
		max_steps=-1,
		num_train_epochs=args.epochs if not args.do_sanity_check else 1,

		per_device_train_batch_size=per_device_batch_size,
		gradient_accumulation_steps=gas_steps,

		save_strategy="no",
		logging_steps=steps_per_epoch,
		logging_strategy="epoch",
		eval_strategy="epoch",
		greater_is_better=False,

		dataset_kwargs={'skip_prepare_dataset': True},
		remove_unused_columns=False,
		report_to=[],
	)

	trainer = SFTTrainer(
	# trainer = SFTSceneInstructionTrainer(
		model=model,
		train_dataset=dataset_train,
		eval_dataset=dataset_val,
		peft_config=peft_config,
		processing_class=tokenizer,
		data_collator=data_collator,
		args=sft_config,
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
		is_sft_training=True,
	))

	init_wandb(args, accelerator)

	if accelerator.is_main_process:
		print("===============================================================================================")
		print(f"RUN ID (wandb): {args.run_id}")
		print(f"JOB ID (sherlock): {args.jid}")
		print(f"number of processes (accelerate): {accelerator.num_processes}")
		if args.use_lora:
			print(f"lora rank: {lora_rank}")
			print(f"lora alpha: {lora_alpha}")
			model.print_trainable_parameters()
		else:
			print("full number of params to train:", model.num_parameters())
		print(f"len of training dataset: {len(trainer.train_dataset)}")
		print(f"samples_per_step: {samples_per_step}")
		print(f"steps_per_epoch: {steps_per_epoch}")
		print(f"total_training_steps: {total_training_steps}")
		print("===============================================================================================")
		print("start training...")

	accelerator.wait_for_everyone()
	trainer.train()

	if accelerator.is_main_process:
		print("SFT TRAINING FINISHED !")