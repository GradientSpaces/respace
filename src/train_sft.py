import torch
from trl import SFTTrainer, SFTConfig
from torch.utils.data import DataLoader
import math
from peft import get_peft_model

from utils import init_wandb, get_lora_config
from train import CustomTrainerCallback
from utils import remove_and_recreate_folder
from src.dataset import SFTSceneDataCollator, load_train_val_test_datasets, WeightedRandomSampler

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

def run_sft_training(model, model_id, max_seq_length, tokenizer, accelerator, dataset_train, dataset_val, dataset_test, sampling_engine, dvc, args):
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