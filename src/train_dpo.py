import json
import threading
from concurrent.futures import ThreadPoolExecutor
from trl import OnlineDPOTrainer, BasePairwiseJudge, OnlineDPOConfig
from trl.trainer.utils import empty_cache
import torch
import torch.nn.functional as F
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
from vllm import SamplingParams
from accelerate import PartialState
from accelerate.utils import broadcast_object_list, gather, gather_object
import numpy as np

from utils import safe_parse_scene, inherit_props_by_id, get_sft_model, init_wandb, create_vllm_engine, destroy_vllm_engine, is_high_quality_sample
from dataset import process_scene_sample, format_with_chat_template, create_full_scene_from_before_and_added
from eval import eval_scene_before_after_with_delta, compute_dss_score, compute_size_l2_dist
from train import CustomTrainerCallback

class OnlineDPOSceneTrainer(OnlineDPOTrainer):
	def __init__(self, *args, **kwargs):

		# self.processing_class = kwargs.pop("processing_class")
		self.max_seq_length = kwargs.pop("max_seq_length")

		self.do_augm = kwargs.pop("do_augm")
		self.do_simple_descs = kwargs.pop("do_simple_descs")
		self.do_full_sg_outputs = kwargs.pop("do_full_sg_outputs")
		self.all_prompts = json.load(open(os.getenv("PTH_ASSETS_METADATA_PROMPTS")))
		self.all_assets_metadata_simple_descs = json.load(open(os.getenv("PTH_ASSETS_METADATA_SIMPLE_DESCS")))

		self.n_samples_per_epoch = kwargs.pop("n_samples_per_epoch", None)

		self.vllm_device = kwargs.pop("vllm_device")
		self.vllm_max_model_len = kwargs.pop("vllm_max_model_len")
		self.vllm_gpu_memory_utilization = kwargs.pop("vllm_gpu_memory_utilization")

		self.use_vllm = kwargs.pop("use_vllm")

		# TRL 0.15's OnlineDPOTrainer natively supports vLLM: when args.use_vllm=True,
		# the parent __init__ creates self.llm = LLM(...) and self.generation_config,
		# and the parent's training_step calls self._generate_vllm() which:
		#   1) syncs the latest training weights into self.llm via load_weights()
		#   2) generates 2 completions per prompt via self.llm.generate()
		#
		# We CANNOT let the parent create self.llm, because:
		#   - It uses gpu_memory_utilization=0.55 on the same GPU as training → OOM
		#   - It has no CUDA determinism workarounds (cumsum_cuda_kernel crash)
		#   - It lacks max_model_len, enable_prefix_caching, separate-GPU placement
		#
		# Strategy: pass use_vllm=False to parent so it skips LLM() creation, then
		# restore args.use_vllm=True so _generate_vllm is called during training_step,
		# and create self.llm ourselves with create_vllm_engine on a dedicated GPU.
		if self.use_vllm:
			gc.collect()
			torch.cuda.empty_cache()
			time.sleep(3.0)

			# Trick: tell parent not to create its own vLLM engine
			kwargs['args'].use_vllm = False
			super().__init__(*args, **kwargs)
			# Restore so parent's training_step will use _generate_vllm path
			kwargs['args'].use_vllm = True

			if self.accelerator.is_main_process:
				gc.collect()
				torch.cuda.empty_cache()
				time.sleep(3.0)

				# Create vLLM on a separate GPU with all determinism workarounds
				self.llm = create_vllm_engine(
					self.model.name_or_path,
					self.vllm_max_model_len,
					gpu_memory_utilization=self.vllm_gpu_memory_utilization,
					device=self.vllm_device,
				)

				self.generation_config = SamplingParams(
					n=2,  # 2 generations per prompt (TRL expectation)
					max_tokens=self.args.max_new_tokens,
					temperature=self.args.temperature,
					top_k=50,
					top_p=0.95,
					detokenize=False,
				)

				print("vLLM setup done on dedicated GPU!")

			self.accelerator.wait_for_everyone()
		else:
			super().__init__(*args, **kwargs)

	def get_train_dataloader(self):
		"""Override to randomly subsample n_samples_per_epoch scenes each epoch."""
		if self.n_samples_per_epoch is not None and self.n_samples_per_epoch < len(self.train_dataset):
			subset = self.train_dataset.shuffle().select(range(self.n_samples_per_epoch))
			print(f"[DPO] Sampled {self.n_samples_per_epoch}/{len(self.train_dataset)} training scenes for this epoch")
			# Temporarily swap train_dataset so the parent builds the dataloader from the subset
			original_dataset = self.train_dataset
			self.train_dataset = subset
			dataloader = super().get_train_dataloader()
			self.train_dataset = original_dataset
			return dataloader
		return super().get_train_dataloader()

	def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
		"""
		Intercept the inputs before they reach the original prediction_step to ensure
		they're in the correct format.
		"""
		# Check if inputs need preprocessing
		if isinstance(inputs, dict) and "prompt" not in inputs:
			inputs, _ = self._preprocess_inputs_for_dpo(inputs)
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
		processed_inputs, gt_objects = self._preprocess_inputs_for_dpo(inputs)
		# Stash GT objects so the judge can access them without modifying the prompt.
		self._current_gt_objects = gt_objects

		# --- Inlined from OnlineDPOTrainer.training_step with HQ masking ---
		model.train()
		prompts = processed_inputs["prompt"]
		batch_size = len(prompts)

		if self.args.use_vllm:
			prompt_ids, prompt_mask, completion_ids, completion_mask = self._generate_vllm(model, prompts)
		else:
			prompt_ids, prompt_mask, completion_ids, completion_mask = self._generate(model, prompts)

		contain_eos_token = torch.any(completion_ids == self.processing_class.eos_token_id, dim=-1)

		logprobs = self._forward(model, prompt_ids, prompt_mask, completion_ids, completion_mask)
		with torch.no_grad():
			if self.ref_model is not None:
				ref_logprobs = self._forward(self.ref_model, prompt_ids, prompt_mask, completion_ids, completion_mask)
			else:
				with self.model.disable_adapter():
					ref_logprobs = self._forward(self.model, prompt_ids, prompt_mask, completion_ids, completion_mask)

		device = logprobs.device

		# Use the judge to compare completions and get HQ labels
		completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
		ranks_of_first_completion = self.judge.judge(
			prompts, list(zip(completions[:batch_size], completions[batch_size:]))
		)
		mask = torch.tensor([rank == 0 for rank in ranks_of_first_completion], device=device)
		hq_mask = torch.tensor(
			[label == "HQ" for label in self.judge.last_labels], dtype=torch.float32, device=device
		)

		batch_range = torch.arange(batch_size, device=device)
		chosen_indices = batch_range + (~mask * batch_size)
		rejected_indices = batch_range + (mask * batch_size)

		cr_indices = torch.cat((chosen_indices, rejected_indices), dim=0)
		cr_logprobs = logprobs[cr_indices]
		cr_ref_logprobs = ref_logprobs[cr_indices]

		padding_mask = ~completion_mask.bool()
		cr_padding_mask = padding_mask[cr_indices]

		cr_logprobs_sum = (cr_logprobs * ~cr_padding_mask).sum(1)
		cr_ref_logprobs_sum = (cr_ref_logprobs * ~cr_padding_mask).sum(1)

		chosen_logprobs_sum, rejected_logprobs_sum = torch.split(cr_logprobs_sum, batch_size)
		chosen_ref_logprobs_sum, rejected_ref_logprobs_sum = torch.split(cr_ref_logprobs_sum, batch_size)
		pi_logratios = chosen_logprobs_sum - rejected_logprobs_sum
		ref_logratios = chosen_ref_logprobs_sum - rejected_ref_logprobs_sum

		logits = pi_logratios - ref_logratios

		if self.args.loss_type == "sigmoid":
			losses = -F.logsigmoid(self.beta * logits)
		elif self.args.loss_type == "ipo":
			losses = (logits - 1 / (2 * self.beta)) ** 2
		else:
			raise NotImplementedError(f"invalid loss type {self.args.loss_type}")

		# Mask out random/tie pairs — only HQ-signal pairs contribute to loss
		n_hq = hq_mask.sum().clamp(min=1)
		loss = (losses * hq_mask).sum() / n_hq
		# loss = (losses * hq_mask).sum() / batch_size

		# Log stats (same as parent)
		self.stats["val/contain_eos_token"].append(contain_eos_token.float().mean().item())
		self.stats["logps/chosen"].append(self.accelerator.gather_for_metrics(chosen_logprobs_sum).mean().item())
		self.stats["logps/rejected"].append(self.accelerator.gather_for_metrics(rejected_logprobs_sum).mean().item())

		kl = logprobs - ref_logprobs
		mean_kl = kl.sum(1).mean()
		self.stats["objective/kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
		non_score_reward = (-self.beta * kl).sum(1)
		mean_non_score_reward = non_score_reward.mean()
		self.stats["objective/non_score_reward"].append(
			self.accelerator.gather_for_metrics(mean_non_score_reward).mean().item()
		)
		mean_entropy = -logprobs.sum(1).mean()
		self.stats["objective/entropy"].append(self.accelerator.gather_for_metrics(mean_entropy).mean().item())
		chosen_rewards = self.beta * (chosen_logprobs_sum - chosen_ref_logprobs_sum)
		gathered_chosen_rewards = self.accelerator.gather_for_metrics(chosen_rewards)
		self.stats["rewards/chosen"].append(gathered_chosen_rewards.mean().item())
		rejected_rewards = self.beta * (rejected_logprobs_sum - rejected_ref_logprobs_sum)
		gathered_rejected_rewards = self.accelerator.gather_for_metrics(rejected_rewards)
		self.stats["rewards/rejected"].append(gathered_rejected_rewards.mean().item())
		margin = gathered_chosen_rewards - gathered_rejected_rewards
		self.stats["rewards/margins"].append(margin.mean().item())
		accuracy = margin > 0
		self.stats["rewards/accuracies"].append(accuracy.float().mean().item())
		self.stats["beta"].append(self.beta)

		if (
			self.args.torch_empty_cache_steps is not None
			and self.state.global_step % self.args.torch_empty_cache_steps == 0
		):
			empty_cache()

		kwargs = {}
		if self.args.n_gpu > 1:
			loss = loss.mean()

		self.accelerator.backward(loss, **kwargs)

		return loss.detach() / self.args.gradient_accumulation_steps

	def _preprocess_inputs_for_dpo(self, inputs):
		num_samples = len(next(iter(inputs.values())))
		samples = []
		for i in range(num_samples):
			sample = {key: values[i] for key, values in inputs.items()}
			samples.append(sample)

		processed_inputs = {"prompt": []}
		gt_objects = []
		for sample in samples:
			full_sample_instr, _, _, processed_sample = process_scene_sample(sample, self.processing_class, self.max_seq_length, self.all_prompts, self.all_assets_metadata_simple_descs, self.do_simple_descs, self.do_augm, self.do_full_sg_outputs)
			formatted_prompt = format_with_chat_template(self.processing_class, full_sample_instr)
			processed_inputs["prompt"].append(formatted_prompt)
			# Keep the GT output object for the judge (not embedded in the prompt)
			sg_output_add = processed_sample.get("sg_output_add_with_jids", processed_sample.get("sg_output_add", ""))
			gt_objects.append(json.loads(sg_output_add) if isinstance(sg_output_add, str) else sg_output_add)

		return processed_inputs, gt_objects

def _eval_completion(obj_prompt, completion, scene_before_with_assets, sampling_engine, n_best_of_n_assets, gpu_lock):
	"""Score a single completion.
	Returns (delta_pbl_loss, txt_pms_sampled_score, scene_after_with_assets) or None on failure.
	"""
	try:
		scene_after = safe_parse_scene(completion.replace('\\"', '"'))
		if scene_after is None:
			return None
		if scene_after.get("objects") is None:
			scene_after = create_full_scene_from_before_and_added(scene_before_with_assets, scene_after)
		if gpu_lock is not None:
			with gpu_lock:
				scene_after_with_assets = sampling_engine.sample_last_asset(
					scene_after, is_greedy_sampling=(True if n_best_of_n_assets == 1 else False)
				)
				inherit_props_by_id(scene_before_with_assets, scene_after_with_assets)
		else:
			scene_after_with_assets = sampling_engine.sample_last_asset(
				scene_after, is_greedy_sampling=(True if n_best_of_n_assets == 1 else False)
			)
			inherit_props_by_id(scene_before_with_assets, scene_after_with_assets)
		# add prompt
		scene_after_with_assets["objects"][-1]["prompt"] = obj_prompt
		# eval scene
		metrics = eval_scene_before_after_with_delta(scene_before_with_assets, scene_after_with_assets)
		return metrics["delta_pbl_loss"], metrics["txt_pms_sampled_score"], scene_after_with_assets
	except Exception as e:
		print(e)
		traceback.print_exc()
		return None

def _judge_single_pair(args_tuple):
	"""Worker for judging one (prompt, completion_pair) — used by ThreadPoolExecutor in SceneQualityJudge.judge()."""
	prompt, completion0, completion1, gt_obj, sampling_engine, n_best_of_n_assets, gpu_lock = args_tuple

	try:
		scene_before = json.loads(re.search(r'<scenegraph>(.*?)</scenegraph>', prompt, re.DOTALL).group(1))

		if gpu_lock is not None:
			with gpu_lock:
				scene_before_with_assets = sampling_engine.sample_all_assets(
					scene_before, is_greedy_sampling=(True if n_best_of_n_assets == 1 else False)
				)
		else:
			scene_before_with_assets = sampling_engine.sample_all_assets(
				scene_before, is_greedy_sampling=(True if n_best_of_n_assets == 1 else False)
			)

		obj_prompt = re.search(r'<add>(.*?)</add>', prompt, re.DOTALL).group(1)

		r0 = _eval_completion(obj_prompt, completion0, scene_before_with_assets, sampling_engine, n_best_of_n_assets, gpu_lock)
		r1 = _eval_completion(obj_prompt, completion1, scene_before_with_assets, sampling_engine, n_best_of_n_assets, gpu_lock)

		# If one completion is unparseable, the other wins immediately
		if r0 is None and r1 is None:
			return (random.choice([0, 1]), "-")
		elif r0 is None:
			return (1, "-")
		elif r1 is None:
			return (0, "-")

		# Use is_high_quality_sample() with real dss and size_l2_dist, exactly as in train_rej.py
		p0 = _score_with_gt(r0, gt_obj, sampling_engine)
		p1 = _score_with_gt(r1, gt_obj, sampling_engine)

		if p0 and not p1:
			return (0, "HQ")
		elif p1 and not p0:
			return (1, "HQ")
		else:
			# Both pass or both fail — no clear signal, pick randomly
			return (random.choice([0, 1]), "-")

	except Exception as e:
		print(f"_judge_single_pair error: {e}")
		traceback.print_exc()
		return (random.choice([0, 1]), "-")

def _score_with_gt(r, gt_obj, sampling_engine):
	"""Compute is_high_quality_sample with real dss and size_l2_dist against GT object."""
	delta_pbl_loss, txt_pms_sampled_score, scene_after_with_assets = r
	new_obj = scene_after_with_assets["objects"][-1]
	new_obj_desc = new_obj.get("desc", "")
	txt_dss_score = compute_dss_score(new_obj_desc, gt_obj.get("desc"), sampling_engine)
	size_l2_dist = compute_size_l2_dist(new_obj.get("size"), gt_obj.get("size"))
	return is_high_quality_sample(delta_pbl_loss, txt_pms_sampled_score, size_l2_dist, txt_dss_score, desc=new_obj_desc)

class SceneQualityJudge(BasePairwiseJudge):
	def __init__(self, sampling_engine, trainer, n_best_of_n_assets=1, use_vlm=False, num_workers=1):
		self.sampling_engine = sampling_engine
		self.trainer = trainer
		self.n_best_of_n_assets = n_best_of_n_assets
		self.use_vlm = use_vlm
		self.num_workers = num_workers
		self.last_labels = []

	def judge(self, prompts, completion_pairs):

		gt_objects = getattr(self.trainer, '_current_gt_objects', [None] * len(prompts))

		gpu_lock = threading.Lock() if self.num_workers > 1 else None

		work_items = [
			(prompt, c0, c1, gt_obj, self.sampling_engine, self.n_best_of_n_assets, gpu_lock)
			for prompt, (c0, c1), gt_obj in zip(prompts, completion_pairs, gt_objects)
		]

		if self.num_workers > 1:
			with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
				raw = list(pool.map(_judge_single_pair, work_items))
		else:
			raw = [_judge_single_pair(item) for item in work_items]

		results = [winner for winner, _ in raw]
		labels = [label for _, label in raw]
		self.last_labels = labels
		print(f"judge results — {results} — {labels}")
		return results

def run_dpo_training(model_id, tokenizer, accelerator, dataset_train, dataset_val, dataset_test, sampling_engine, dvc, args):

	accelerator.wait_for_everyone()
	gc.collect()
	torch.cuda.empty_cache()

	model, max_seq_length, lora_rank, lora_alpha = get_sft_model(model_id, args, accelerator)

	# Number of scenes to sample per epoch (randomly drawn from full dataset each epoch)
	n_samples_per_epoch = getattr(args, 'dpo_samples_per_epoch', 512)
	# n_samples_per_epoch = getattr(args, 'dpo_samples_per_epoch', 1024)

	# Fix issue with models that ignore padding left
	tokenizer.padding_side = "left"
	if "qwen" in args.llm:
		model.config.use_cache = False

	num_processes = accelerator.num_processes

	print("num_processes: ", num_processes)

	samples_per_step = args.dvc_batch_size * num_processes * args.gas_steps
	steps_per_epoch = math.ceil(n_samples_per_epoch / samples_per_step)
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
		
		# lr_scheduler_type="cosine",
		
		lr_scheduler_type="constant_with_warmup",
		warmup_steps=5,

		# DPO specific parameters
		beta=args.dpo_beta,
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

	num_eval_workers = getattr(args, 'num_eval_workers', 1)

	# Create judge with trainer=None first (set after trainer construction)
	judge = SceneQualityJudge(
		sampling_engine=sampling_engine,
		trainer=None,
		n_best_of_n_assets=1,
		use_vlm=False,
		num_workers=num_eval_workers,
	)

	trainer = OnlineDPOSceneTrainer(
		model=model,
		# tokenizer=tokenizer,
		max_seq_length=max_seq_length,
		judge=judge,
		args=dpo_config,
		processing_class=tokenizer,
		train_dataset=dataset_train,
		eval_dataset=dataset_val.select(range(4)),
		do_augm=args.do_augm,
		do_simple_descs=args.do_simple_descs,
		do_full_sg_outputs=args.do_full_sg_outputs,

		n_samples_per_epoch=n_samples_per_epoch,
		use_vllm=args.use_vllm,
		vllm_max_model_len=max_seq_length,

		# vllm_device="auto",
		# vllm_gpu_memory_utilization=0.15,

		# vllm_device=f"cuda:{num_processes}" if args.use_vllm else "auto",

		vllm_device="cuda:1",
		vllm_gpu_memory_utilization=0.8,

		# vllm_device="cuda:0",
		# vllm_gpu_memory_utilization=0.2,
	)

	# Now wire the judge to the trainer so it can read _current_gt_objects
	judge.trainer = trainer

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
		print(f"n_samples_per_epoch: {n_samples_per_epoch}")
		print(f"per device batch size: {args.dvc_batch_size}")
		print(f"samples_per_step: {samples_per_step}")
		print(f"steps_per_epoch: {steps_per_epoch}")
		print(f"log_every_n_steps: {log_every_n_steps}")
		print(f"total_training_steps: {total_training_steps}")
		print(f"using vLLM: {args.use_vllm}")
		print(f"number of eval workers: {num_eval_workers}")
		print("===============================================================================================")

	accelerator.wait_for_everyone()
	trainer.train()

	if accelerator.is_main_process:
		print("ONLINE DPO TRAINING FINISHED!")