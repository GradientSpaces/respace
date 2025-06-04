import setuptools
import os
import torch
import argparse
import numpy  as np
import json
import logging
import sys
from dotenv import load_dotenv
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import Sampler
from safetensors.torch import load_file
from peft import get_peft_model
import wandb
import pdb

from src.dataset import load_train_val_test_datasets, count_samples_exceeding_max_length, count_samples_testset_seeds_exceeding_max_length
from src.sample import AssetRetrievalModule
from src.test import run_test
from src.utils import set_seeds, get_model, remove_and_recreate_folder, init_wandb, get_lora_config, StreamToLogger, get_test_instrs_all
from src.train_grpo import run_grpo_training
from src.train_dpo import run_dpo_training
from src.train_sft import run_sft_training

# **********************************************************************************************************

def main(args):
	load_dotenv(args.env)

	if args.use_gpu:
		torch.cuda.empty_cache()
		torch.cuda.init()

	if args.test_ckpt is None and args.jid is None:
		raise ValueError("--jid must be provided for training")
	
	if args.resume and args.jid is None:
		raise ValueError("cannot resume without specifying --jid")
	
	if args.resume:
		resume_checkpoint_path = f"./ckpts/{args.jid}/checkpoint-last"
		if not os.path.exists(resume_checkpoint_path):
			raise ValueError(f"cannot resume: checkpoint {resume_checkpoint_path} does not exist")
		print(f"resuming training from {resume_checkpoint_path}")

	accelerator = None
	accelerator = Accelerator()
	dvc = accelerator.device
	print(f"idx [{accelerator.process_index}] using jid: {args.jid}")

	if args.use_logfile:
		print(f"switching to logfile in the folder ./logs/{args.jid}")
		jid = args.jid
		os.makedirs(f"./logs/{jid}", exist_ok=True)
		logging.basicConfig(format=f"%(asctime)s — [device {torch.cuda.current_device()}] — %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p — ", filename=f"./logs/{jid}/main.log", filemode="a", level=logging.INFO)
		sys.stdout = StreamToLogger(logging.getLogger('stdout'), dvc)
		sys.stderr = StreamToLogger(logging.getLogger('stderr'), dvc, log_level=logging.ERROR)

	is_sft_training = (not args.do_grpo and not args.do_dpo)
	use_determ = (True if is_sft_training else False)
	set_seeds(args.seed, use_determ=use_determ) #, use_deterministic=use_deterministic)
	print(f"After set_seeds, deterministic algorithms enabled: {torch.are_deterministic_algorithms_enabled()}, use_deterministic was {use_determ}")

	print(f"using env file: {args.env}")
	print(f"cuda available" if dvc != "auto" else "cuda NOT available")
	print(f"process running on device: {accelerator.device}")

	dataset_train, dataset_val, dataset_test = load_train_val_test_datasets(args.lambda_instr_exp, args.use_cached_dataset, args.room_type, args.do_sanity_check, args.seed, accelerator)
	if args.prep_data_only:
		print("len of datasets:")
		print(f"train: {len(dataset_train)}")
		print(f"val: {len(dataset_val)}")
		print(f"test: {len(dataset_test)}")
		# all_test_instrs = get_test_instrs_all(args.room_type)
		# pdb.set_trace()
		exit()

	if args.test_ckpt is not None:
		model_id = f"./ckpts/{args.test_ckpt}"
	else:
		if args.llm == "llama-3.2-1B":
			model_id = "meta-llama/Llama-3.2-1B-Instruct"
		elif args.llm == "llama-3.2-3B":
			model_id = "meta-llama/Llama-3.2-3B-Instruct"
		elif args.llm == "llama-3.2-7B":
			model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
		elif args.llm == "qwen-2.5-0.5B":
			model_id = "Qwen/Qwen2.5-0.5B-Instruct"
		elif args.llm == "qwen-2.5-1.5B":
			model_id = "Qwen/Qwen2.5-1.5B-Instruct"
		elif args.llm == "qwen-2.5-math-1.5B":
			model_id = "Qwen/Qwen2.5-Math-1.5B-Instruct"
		else:
			raise ValueError("model not found")

	model, tokenizer, max_seq_length = get_model(model_id, args.use_gpu, accelerator)

	# print("\nChecking for samples exceeding max_seq_length after processing:")
	# all_prompts = json.load(open(os.getenv("PTH_ASSETS_METADATA_PROMPTS")))
	# all_assets_metadata_simple_descs = json.load(open(os.getenv("PTH_ASSETS_METADATA_SIMPLE_DESCS")))
	# for split_name, dataset_split in [("train", dataset_train), ("val", dataset_val), ("test", dataset_test)]:
	# 	count_samples_exceeding_max_length(dataset_split, tokenizer, max_seq_length, all_prompts, all_assets_metadata_simple_descs, args.do_simple_descs, args.do_full_sg_outputs)
	# # all_test_instrs = get_test_instrs_all(args.room_type)
	# # count_samples_testset_seeds_exceeding_max_length(dataset_test, tokenizer, max_seq_length, all_test_instrs, all_prompts, all_assets_metadata_simple_descs, args.do_simple_descs, args.do_full_sg_outputs)	
	# exit()

	sampling_engine = AssetRetrievalModule(lambd=0.5, sigma=0.05, temp=0.2, top_p=0.95, top_k=20, asset_size_threshold=0.5, rand_seed=args.seed, accelerator=accelerator, do_print=False, is_sft_training=is_sft_training)

	# n_max = 32
	# all_prompts = json.load(open(os.getenv("PTH_ASSETS_METADATA_PROMPTS")))
	# all_assets_metadata_simple_descs = json.load(open(os.getenv("PTH_ASSETS_METADATA_SIMPLE_DESCS")))
	# run_test(model, tokenizer, accelerator, dvc, "test", args.room_type, dataset_test.select(range(n_max)), max_seq_length, sampling_engine, all_prompts, all_assets_metadata_simple_descs, args.do_simple_descs, args, do_print=False)
	# run_test(model, tokenizer, accelerator, dvc, "train", args.room_type, dataset_train.select(range(n_max)), max_seq_length, sampling_engine, all_prompts, all_assets_metadata_simple_descs, args.do_simple_descs, args, do_print=False)
	# exit()

	if args.do_dpo:
		run_dpo_training(model_id, tokenizer, accelerator, dataset_train, dataset_val, dataset_test, sampling_engine, dvc, args)
	elif args.do_grpo:
		run_grpo_training(model_id, tokenizer, accelerator, dataset_train, dataset_val, dataset_test, sampling_engine, dvc, args)
	elif args.test_ckpt is None:
		run_sft_training(model, model_id, max_seq_length, tokenizer, accelerator, dataset_train, dataset_val, dataset_test, sampling_engine, dvc, args)
	elif args.do_run_interactive:
		run_interactive(model, tokenizer, max_seq_length, accelerator, dvc, dataset_train, dataset_val, dataset_test, sampling_engine, args)
	else:
		init_wandb(args, accelerator, resume_id=(args.test_ckpt.split("/")[0]))
		all_prompts = json.load(open(os.getenv("PTH_ASSETS_METADATA_PROMPTS")))
		all_assets_metadata_simple_descs = json.load(open(os.getenv("PTH_ASSETS_METADATA_SIMPLE_DESCS")))
		n_max = 500
		if args.do_testset_only:
			run_test(model, tokenizer, accelerator, dvc, "test", args.room_type, dataset_test.select(range(n_max)), max_seq_length, sampling_engine, all_prompts, all_assets_metadata_simple_descs, args.do_simple_descs, args, do_print=False)
		else:
			run_test(model, tokenizer, accelerator, dvc, "train", args.room_type, dataset_train.select(range(n_max)), max_seq_length, sampling_engine, all_prompts, all_assets_metadata_simple_descs, args.do_simple_descs, args, do_print=False)
			run_test(model, tokenizer, accelerator, dvc, "val", args.room_type, dataset_val.select(range(n_max)), max_seq_length, sampling_engine, all_prompts, all_assets_metadata_simple_descs, args.do_simple_descs, args, do_print=False)
			run_test(model, tokenizer, accelerator, dvc, "test", args.room_type, dataset_test.select(range(n_max)), max_seq_length, sampling_engine, all_prompts, all_assets_metadata_simple_descs, args.do_simple_descs, args, do_print=False)
		
# **********************************************************************************************************

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Author: Martin Juan José Bucher')

	parser.add_argument('--jid', type=str)
	parser.add_argument('--run-id', type=str)
	parser.add_argument('--test-ckpt', type=str)
	parser.add_argument('--env', type=str, default=".env")

	parser.add_argument('--use-cached-dataset', action='store_true', default=False)
	parser.add_argument('--use-wandb', action='store_true', default=False, help='enable logging via wandb')
	parser.add_argument('--use-logfile', action='store_true', default=False)
	parser.add_argument('--use-gpu', action='store_true', default=False)

	parser.add_argument('--seed', type=int, default=1234, help="random seed for reproducibility")
	
	parser.add_argument('--dvc-batch-size', type=int, default=8)
	parser.add_argument('--gas-steps', type=int, default=4)

	parser.add_argument('--epochs', type=int, default=50)
	parser.add_argument('--use-lora', action='store_true', default=False)
	parser.add_argument('--lora-rank', type=int)
	parser.add_argument('--lora-alpha', type=int)
	parser.add_argument('--do-augm', action='store_true', default=False)
	
	parser.add_argument('--llm', type=str, choices=["llama-3.2-1B", "llama-3.2-3B", "llama-3.2-7B", "qwen-2.5-0.5B", "qwen-2.5-1.5B"], default="llama-3.2-1B")
	
	parser.add_argument('--do-grpo', action='store_true', default=False)
	parser.add_argument('--grpo-num-gen', type=int, default=8)
	parser.add_argument('--do-dpo', action='store_true', default=False)
	parser.add_argument('--do-token-masking', action='store_true', default=False)
	parser.add_argument('--grpo-do-neutral-rewards', action='store_true', default=False)
	parser.add_argument('--grpo-reward-version', type=str, default="v1")
	parser.add_argument('--grpo-beta-kl', type=float, default=0.0)
	parser.add_argument('--grpo-temp', type=float, default=0.7)
	
	parser.add_argument('--grpo-learning-rate', type=float, default=1e-6)
	parser.add_argument('--dpo-learning-rate', type=float, default=1e-7)

	parser.add_argument('--use-vllm', action='store_true', default=False)

	parser.add_argument('--room-type', type=str, choices=["bedroom", "livingroom", "all"], default="bedroom")
	parser.add_argument('--do-simple-descs', action='store_true', default=False)
	parser.add_argument('--do-full-sg-outputs', action='store_true', default=False)
	parser.add_argument('--do-renderings', action='store_true', default=False)
	parser.add_argument('--lambda-instr-exp', type=float, default=0.0)

	parser.add_argument('--n-best-of-n-llm', type=int, default=1)
	
	parser.add_argument('--do-sanity-check', action='store_true', default=False)
	parser.add_argument('--prep-data-only', action='store_true', default=False)
	parser.add_argument('--test-bs', type=int, default=1)
	parser.add_argument('--do-testset-only', action='store_true', default=False)
	parser.add_argument('--do-run-interactive', action='store_true', default=False)

	parser.add_argument('--vllm-host', type=str)
	parser.add_argument('--vllm-port', type=str)

	parser.add_argument('--resume', action='store_true', default=False)

	main(parser.parse_args())