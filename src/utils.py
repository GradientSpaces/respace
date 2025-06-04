import torch
import numpy as np
import random
import os
from accelerate.utils import set_seed
from shapely.geometry import Polygon
import shutil
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from cleanfid import fid
import hashlib
import pickle
import traceback
import wandb
import time
import logging
from cleanfid.clip_features import CLIP_fx, img_preprocess_clip
import shapely
import trimesh

def get_tgseed(seed):
	g = torch.Generator()
	g.manual_seed(seed)
	return g

def set_seeds(seed, use_determ=True):
	# print("setting random seeds...")
	torch.manual_seed(seed)
	random.seed(seed)
	np.random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	torch.cuda.manual_seed_all(seed)
	torch.cuda.manual_seed(seed)
	
	if use_determ:
		torch.backends.cudnn.deterministic = True
		torch.use_deterministic_algorithms(True)
	else:
		torch.backends.cudnn.deterministic = False
		torch.use_deterministic_algorithms(False)
	
	# torch.backends.cudnn.benchmark = True
	#if 'CUBLAS_WORKSPACE_CONFIG' in os.environ:
		# del os.environ['CUBLAS_WORKSPACE_CONFIG']
	set_seed(seed) # HF accelerate

def get_pth_mesh(asset_jid):
	return os.path.join(os.getenv("PTH_3DFUTURE_ASSETS"), asset_jid, "raw_model.glb")

def create_floor_plan_polygon(bounds):
	return Polygon(np.array(bounds)[:, [0, 2]].tolist())

def remove_and_recreate_folder(pth):
	if os.path.exists(pth):
		shutil.rmtree(pth, ignore_errors=False)
	os.makedirs(pth, exist_ok=True)

def get_llama_vanilla_pipeline():
	model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
	pipeline = transformers.pipeline(
		"text-generation",
		model=model_id,
		model_kwargs={
			"torch_dtype": torch.bfloat16,
		},
		device_map="auto"
	)
	return pipeline

def precompute_fid_scores_for_caching(fid_score_name, pth_dataset):
	if fid.test_stats_exists(fid_score_name, mode="clean"):
		fid.remove_custom_stats(fid_score_name, mode="clean")
	fid.make_custom_stats(fid_score_name, pth_dataset, mode="clean")
	
	if fid.test_stats_exists(fid_score_name, model_name="clip_vit_b_32", mode="clean"):
		fid.remove_custom_stats(fid_score_name, model_name="clip_vit_b_32", mode="clean")
	fid.make_custom_stats(fid_score_name, pth_dataset, model_name="clip_vit_b_32", mode="clean")

def compute_fid_scores(fid_prefix, fid_score_name, pth_src, pth_gen, do_renderings, aggregated_metrics, dataset_res=1024):

	if do_renderings == False or (not os.path.exists(pth_gen)) or (len(os.listdir(pth_gen)) < 2):
		print("skipping FID computation")
		aggregated_metrics[f"fid_score_{fid_prefix}"] = float('inf')
		aggregated_metrics[f"fid_clip_score_{fid_prefix}"] = float('inf')
		aggregated_metrics[f"kid_score_{fid_prefix}"] = float('inf')
		return

	if not fid.test_stats_exists(fid_score_name, "clean"):
		precompute_fid_scores_for_caching(fid_score_name, pth_src)

	fid_score = fid.compute_fid(pth_gen, dataset_name=fid_score_name, dataset_res=dataset_res, dataset_split="custom")
	fid_clip_score = fid.compute_fid(pth_gen, dataset_name=fid_score_name, dataset_res=dataset_res, model_name="clip_vit_b_32", dataset_split="custom")
	kid_score = fid.compute_kid(pth_gen, dataset_name=fid_score_name, dataset_res=dataset_res, dataset_split="custom")

	aggregated_metrics[f"fid_score_{fid_prefix}"] = round(fid_score, 2)
	aggregated_metrics[f"fid_clip_score_{fid_prefix}"] = round(fid_clip_score, 2)
	aggregated_metrics[f"kid_score_{fid_prefix}"] = round(kid_score / 0.001, 2)

def compute_diversity_score(fid_prefix, pth_gen, do_renderings, dvc, aggregated_metrics):

	if do_renderings == False or (not os.path.exists(pth_gen)) or (len(os.listdir(pth_gen)) < 2):
		print("skipping diversity computation")
		aggregated_metrics[f"diversity_score_{fid_prefix}"] = float('inf')
		return

	model = CLIP_fx("ViT-B/32", device=dvc)
	custom_fn_resize = img_preprocess_clip

	features = fid.get_folder_features(pth_gen, model, device=dvc, mode="clean", custom_fn_resize=custom_fn_resize)

	cov = np.cov(features, rowvar=False)
	diversity_score = np.trace(cov)

	aggregated_metrics[f"diversity_score_{fid_prefix}"] = round(diversity_score, 2)

def get_scene_hash(scene):
	scene_str = json.dumps(scene, sort_keys=True)
	scene_hash = hashlib.md5(scene_str.encode()).hexdigest()
	return scene_hash

def get_pths_dataset_split(room_type, dataset_split, prefix=None):
	pth_base = os.getenv("PTH_STAGE_3") if prefix is None else os.path.join(prefix, os.getenv("PTH_STAGE_3"))
	with open(os.path.join(pth_base, f"{room_type}_splits.pkl"), 'rb') as f:
		all_splits = pickle.load(f)
	return all_splits[dataset_split]

def get_test_instrs_all(room_type):
	with open(os.path.join(os.getenv("PTH_STAGE_3"), f"{room_type}_splits.pkl"), 'rb') as f:
		all_splits = pickle.load(f) 
		return all_splits["test_instrs"]

def inherit_props_by_id(scene_before, scene_after):

	len_before = 0
	if (scene_before.get("objects") is not None) and isinstance(scene_before.get("objects"), list):
		len_before = len(scene_before.get("objects"))

	len_after = 0
	if (scene_after.get("objects") is not None) and isinstance(scene_after.get("objects"), list):
		len_after = len(scene_after.get("objects"))

	if len_after == (len_before + 1):
		for i in range(len_before):
			scene_after['objects'][i]['sampled_asset_jid'] = scene_before['objects'][i]['sampled_asset_jid']
	elif len_after == len_before:
		for i in range(len_before-1):
			scene_after['objects'][i]['sampled_asset_jid'] = scene_before['objects'][i]['sampled_asset_jid']
	else:
		print(f"⛔️ inheriting props: unknown matching lengths, before: {len_before}, after: {len_after}")
		# print("")
		# print(scene_before)
		# print("")
		# print(scene_after)
		# print("")

def get_model(model_id, use_gpu, accelerator=None, do_not_load_hf_model=False):

	print(f"get_model(): loading tokenizer for {model_id}")
	tokenizer = AutoTokenizer.from_pretrained(model_id)

	model_type = ""
	if "ckpts" in model_id:
		config = json.load(open(f"{model_id}/config.json"))
		model_type = config["model_type"]
	elif model_id == "gradient-spaces/respace-sg-llm-1.5b":
		model_type = "qwen2"

	if "Qwen" in model_id or "qwen" in model_type:
		print("setting qwen tokenizer settings...")
		tokenizer.pad_token_id = 151643
	else:
		# llama3.1
		tokenizer.pad_token = '<|finetune_right_pad_id|>'

	# fix max length for all models
	# if args.room_type == "bedroom":
	# 	max_seq_length = 3000
	# elif args.room_type == "livingroom":
	# 	max_seq_length = 3500
	# else:
	# 	max_seq_length = 3800

	max_seq_length = 3000
	# max_seq_length = 2500

	tokenizer.model_max_length = max_seq_length
	
	if use_gpu and accelerator is not None:
		device_map = ({"": accelerator.device})
	elif use_gpu and accelerator is None:
		device_map = "cuda"
	else:
		device_map = "cpu"

	print(f"get_model(): loading model for {model_id}")
	if do_not_load_hf_model == True:
		model = None
	else:
		model = AutoModelForCausalLM.from_pretrained(
			model_id,
			# device_map="cuda",
			device_map=device_map,
			# device_map="auto",
			torch_dtype=torch.bfloat16,
			attn_implementation="flash_attention_2" if use_gpu else "sdpa",
		)

	return model, tokenizer, max_seq_length

def assert_props_for_obj(obj):
	assert obj.get("desc") is not None
	assert isinstance(obj.get("desc"), str)
	assert len(obj.get("desc")) > 0
	
	assert obj.get("pos") is not None
	assert isinstance(obj.get("pos"), list)
	assert len(obj.get("pos")) == 3
	# assert all(isinstance(x, float) for x in obj.get("pos"))

	assert obj.get("rot") is not None
	assert isinstance(obj.get("rot"), list)
	assert len(obj.get("rot")) == 4
	# assert all(isinstance(x, float) for x in obj.get("rot"))

	assert obj.get("size") is not None
	assert isinstance(obj.get("size"), list)
	assert len(obj.get("size")) == 3
	# assert all(isinstance(x, float) for x in obj.get("size"))

def cast_scene_floats(scene_json):
	scene_json["pos"] = [float(x) for x in scene_json["pos"]]
	scene_json["rot"] = [float(x) for x in scene_json["rot"]]
	scene_json["size"] = [float(x) for x in scene_json["size"]]
	return scene_json

def safe_parse_scene(scene_text):
	try:
		scene_json = json.loads(scene_text)

		if scene_json.get("objects") is None:
			assert_props_for_obj(scene_json)
			scene_json = cast_scene_floats(scene_json)
		else:
			for idx, obj in enumerate(scene_json.get("objects")):
				assert_props_for_obj(obj)
				# cast every object to floats
				obj = cast_scene_floats(obj)
				# overwrite existing object in scene_json
				scene_json["objects"][idx] = obj
		return scene_json
	except Exception as exc:
		print(exc)
		print(f"could not parse scene for text: --'{scene_text}'--")
		traceback.print_exc()
		return None
	
def get_room_type_from_id(room_id):
	# room_types = ["bedroom", "livingroom", "diningroom", "library"]
	room_id_lc = room_id.lower()
	if "bedroom" in room_id_lc:
		return "bedroom"
	if "livingroom" in room_id_lc or "livingdiningroom" in room_id_lc or "diningroom" in room_id_lc:
		return "livingroom"
	return "other"

def create_category_lookup(all_assets_metadata_orig, all_assets_metadata):
	jid_to_category = {}
	for item in all_assets_metadata_orig:
		if item.get('category') is not None:
			jid_to_category[item.get('model_id')] = item.get('category').lower().replace(" / ", "/")
		else:
			jid_to_category[item.get('model_id')] = "unknown_category"
	
	desc_to_category = {}
	for jid, metadata in all_assets_metadata.items():
		if jid in jid_to_category:
			desc_to_category[metadata.get('summary')] = jid_to_category[jid]
	
	return desc_to_category

def get_system_prompt_sgllm():
	return "You are a world-class leading interior design expert. Your task is to add furniture given the descriptions in the header and the current list of furniture in the body. You must respond ONLY with a valid JSON string that matches precisely the *format* of the existing JSON in the request. Only output the JSON as a plain string and nothing else."

# """you are a world-class leading interior design expert. your task is to fulfill the request of the user about interior design but you have help of another world-class expert model that can only be called in an XML-style API.

# # your input
# - <prompt> : the user request
# - <scenegraph> : the current scene may be given as a JSON object. in some cases, there will be no scene graph given, which means there is no "current" scene to work with. this argument is optional.

# # your task

# ## adding
# - if the user wants to add one or multiple objects, you create an <add> command for every object/furniture at add it to the list in "commands". for the description, you should refer to the subject with a maximum of five additional descriptive words. the first words should refer to the color / style / shape / etc., while the last word should always be the main subject.
# - if the user request provides an existing scene description provided via <scenegraph>...</scenegraph>, you should try to match the style of the existing objects by providing a similar style as part of the description of your commands. if the user provides a very detailed and long description of what is required, then try to distill it into a more compact description that is still accurate.
# - if the user provides some requirement about particular furniture that should be present in the room, you should always add these objects via <add> commands.

# ## bounds
# - if the user provides a high-level description of a room, e.g. a "bedroom with a king-sized bed", WITHOUT any scenegraph, you add a "<bounds>...</bounds>" command with a one-word description of the room type, given the user requests. available room type choices are: [ 'bedroom', 'livingroom', 'all' ]
# - example: <bounds>bedroom</bounds>
# - the <bounds> command should always come first. 
# - after the bounds command, you can add up to N commands for adding objects via <add>. 
# - if the flag <do_fill_room> is set to true, you should create a few commands that match the user prompt. if the flag is false, you should not add any objects at all via <add>.
# - if the user provides an existing scenegraph, you should never create a <bounds> command.

# ## removing / swapping
# - if the user wants to remove one to multiple objects, you remove the objects that match the request from <scenegraph> and produce a new <scenegraph>
# - if no modification to the existing scenegraph is performed, you MUST leave the <scenegraph> key as null.
# - if the user wants to swap furniture, you remove the objects that match that description and reflect that with the new "scenegraph" key and add the new objects in the list of commands via <add>. we first need to remove an object before we can add a new one so removing them is crucial.

# ## resampling
# - if the user wants to have another 3D asset for a specific object, you add a "<resample>...</resample>" command to the list of commands. 
# - the resample command should contain the EXACT same description of the object that was used for <add></add>.
# - you can only resample objects that are already present in the scene graph. if the object is not present, you should ignore the resample command.
# - if the user wants to resample multiple objects, you can add multiple resample commands to the list of commands.
# - if the user request is ambigious and you can not determine which object should be resampled, you should make your best guess.

# # your output
# - your answer is single JSON object with three keys: "commands", "response", and "scenegraph".
# - the first key is the "commands" key and contains a list of commands in the format specified above.
# - the second key is the "response" key and contains a response in natural language that will be returned to the user. this should be a short but helpful and friendly response in an uplifting tone that is in context of the user request, assuming that the request has been fulfilled.
# - the third key is the "scenegraph" key and contains the (optionally) updated scene graph as a JSON format. you must EXACTLY follow the format of the existing JSON. if you only add objects to the scene (without removing any), you MUST leave the value under the "scenegraph" key as null. this is very important. you MUST NOT provide a value if you did not modify the scenegraph.
# - if the request is completely out-of-context and can not be phrased into a list of commands or is not about interior design, you can leave the "commands" list empty and just add a natural language response that informs the user about the limitations of your system. in that case, you also leave the scenegraph key empty

# your answer is always a SINGLE valid JSON object with three keys: "commands", "answer", and "scenegraph".

# remember:
# (1) if you do not remove any objects from the scene, you should leave the scenegraph as null. 
# (2) ONLY if no scenegraph is provided, you should add the <bounds> command. if you see "bounds_top" in the input, you must not provide any bounds command.
# (3) you always output the final JSON object as a plain string and nothing else. NEVER use markdown."""

# You will see two different renderings of the same scene, one from an axonometric perspective and one from a top-down bird's eye view. It is the same scene but from two different angles.
# - You will assign only one score per criterion, based on BOTH renderings jointly. 
# - Stay consistent in how you judge the scene based on the two images, since they are the same scene from different angles. You must NOT analyze or judge the scene for each image individiually.
# - You MUST NOT assign different scores to the two renderings separately. In your answer, you have only 4 subscores and 1 final score.
# You MUST finish with the exact final sentence "Thus, the final score is:" and your SINGLE-DIGIT final score as an INTEGER.

def get_vlm_prompt(room_type, scenegraph):
	
	prompt = f"""You are a world-class interior design expert and your task is to analyze a two renderings of an indoor scene and choose which one is better.
	
You will see two different versions of the same scene except for the last object of the same object-category that was added.

You will use the following criteria to judge which version is more coherent. All criteria are weighted equally. The criterias are:

- (1) Layout Coherence: Does the general layout of the object arrangements make sense? Does the arrangement of 3D assets adhere to realism and common sense (considering position & orientation)? Consider room boundaries and how furniture is placed within the general floor plan. Also consider intra-object relationships and how well they are placed in relation to each other. If objects overlap slightly or are out-of-bounds, you should give very bad scores (< 3).
- (2) Human-object interaction: Does the layout represent a functional arrangement? Does the arrangement allow for human-based interaction and movement within the space, or is the arrangement bulky and prevents appropriate human interaction? It is VERY important to consider how objects are oriented inside the space.
- (3) Colour scheme, choice of materials and specific asset selection: Does the overall asset/furniture selection make sense? Is the stylistic choice appropriate? Consider the interaction between different styles and how well they work together. You must NOT judge based on the wooden floor since we always take the same floor for all scenes.

Very important to consider:

- You must limit your judgement to the placement of furniture and the choice of specific assets.
- You must NOT criticise lack of detail, lack of personality, or personal touch. The scenes are from a synthetic dataset and they represent simplified indoor scenes.
- The room may be partially furnished and it is very important that your judgement is not based on how well the room is filled or if certain things are missing.
- Do NOT hallucinate and stick to the furntiture as seen in the renderings.
- There are NO doors, windows or walls in the scene. Both scenes have the same floor plan and the same fixed beige wooden floor.

You MUST finish with the exact final sentence "Thus, the final answer is: <x>" where <x> is either A (first image) or B (second image)

You will be provided with the room type and (optionally) with an accompanying list of objects that describe the scene.
	
Room type: '{room_type}'"""
	
	# prompt = "What furniture do you see in two images? They are the same scene from different angles."
	
	if scenegraph:
		prompt += f"\nList of objects{scenegraph}"
	
	return prompt

def init_wandb(args, accelerator, resume_id=None):
	if args.use_wandb and (accelerator.is_main_process or accelerator is None):
		wandb.init(entity="mnbucher", project="stan-24-sgllm", name=args.run_id, id=(resume_id if resume_id is not None else args.jid), resume=("allow" if resume_id is not None else None))
		wandb.config.update(args)

def get_sft_model(model_id, args, accelerator):
	is_lora_model = os.path.exists(f"./ckpts/{args.test_ckpt}/adapter_config.json")

	if is_lora_model:
		print(f"[ idx {accelerator.process_index} ] found LoRA model, loading with PEFT")

		# TODO: DOES NOT WORK YET !!

		model, _, max_seq_length = get_model(f"./ckpts/{args.test_ckpt}", args.use_gpu, accelerator)
		
		# model = model.merge_and_unload()

		# adapter_config = json.load(open(f"./ckpts/{args.test_ckpt}/adapter_config.json"))
		# lora_rank = adapter_config["r"]
		# lora_alpha = adapter_config["lora_alpha"]

		# base_model, _, max_seq_length = get_model(model_id, args, accelerator)

		# peft_config = get_lora_config(lora_rank, lora_alpha)	
		# model = get_peft_model(base_model, peft_config)
		
		# print(f"[ idx {accelerator.process_index} ] prepared get_peft_model (with random weights for lora)")

		# time.sleep(accelerator.process_index * 5)  # 5 second delay per process

		# print(f"[ idx {accelerator.process_index} ] loading peft adapter weights...")
		# torch.cuda.empty_cache()
		# model.load_adapter(
		# 	f"./ckpts/{args.test_ckpt}",
		# 	adapter_name="default",
		# 	# device_map=({"": accelerator.device}),
		# 	# device_map="auto",  # Try auto device mapping
		# 	device_map={"": "cpu"},
		# 	is_trainable=False
		# )

		# print(f"[ idx {accelerator.process_index}] PEFT model loaded successfully")

		return model, max_seq_length, None, None

	else:
		model, _, max_seq_length = get_model(f"./ckpts/{args.test_ckpt}", args.use_gpu, accelerator)

		print(f"[ idx {accelerator.process_index}] dense model loaded successfully")

		return model, max_seq_length, None, None

def get_lora_config(lora_rank, lora_alpha):
	from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
	peft_config = LoraConfig(
		lora_alpha=lora_alpha,
		lora_dropout=0.05,
		r=lora_rank,
		# target_modules = ['embed_tokens', 'lm_head', 'up_proj', 'gate_proj', 'v_proj', 'q_proj', 'k_proj', 'o_proj', 'down_proj'],
		target_modules = ['embed_tokens', 'up_proj', 'gate_proj', 'v_proj', 'q_proj', 'k_proj', 'o_proj', 'down_proj'],
		bias="none",
		task_type="CAUSAL_LM",
	)
	return peft_config

class StreamToLogger(object):
	def __init__(self, logger, dvc, log_level=logging.INFO):
		self.logger = logger
		self.log_level = log_level
		self.linebuf = ''
		self.dvc = dvc

	def write(self, buf):
		for line in buf.rstrip().splitlines():
			self.logger.log(self.log_level, f"[ dvc:{self.dvc} ] — {line.rstrip()}")

	def flush(self):
		pass

	def isatty(self):
		return False