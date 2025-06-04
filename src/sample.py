import json
import numpy as np
from transformers import AutoTokenizer, SiglipTextModel, SiglipTextConfig
import pdb
import pickle
import torch
from matplotlib import pyplot as plt
import uuid
from scipy.special import softmax
import torch
import torch.nn as nn
import numpy as np
import uuid
from dotenv import load_dotenv
import traceback
# from sentence_transformers import SentenceTransformer
import os
import copy


class AssetRetrievalModule(nn.Module):
	def __init__(self, lambd, sigma, temp, top_p, top_k, asset_size_threshold, rand_seed=None, accelerator=None, dvc=None, do_print=False, is_sft_training=False):
		# torch.use_deterministic_algorithms(False)
		super().__init__()

		self.accelerator = accelerator
		self.dvc = dvc

		self.all_assets_metadata = json.load(open(os.getenv("PTH_ASSETS_METADATA")))
		self.all_assets_metadata_scaled = json.load(open(os.getenv("PTH_ASSETS_METADATA_SCALED")))

		# config = SiglipTextConfig.from_pretrained("google/siglip-so400m-patch14-384")
		# config.max_position_embeddings = 64
		# self.siglip_model = SiglipTextModel.from_pretrained("google/siglip-so400m-patch14-384", config=config)
		self.siglip_model = SiglipTextModel.from_pretrained("google/siglip-so400m-patch14-384")
		self.siglip_tokenizer = AutoTokenizer.from_pretrained("google/siglip-so400m-patch14-384")
		
		with open(os.getenv("PTH_ASSETS_EMBED"), 'rb') as fp: 
			model_info_martin_embeds = pickle.load(fp)

		all_embeds = np.array(model_info_martin_embeds.get("embeds"))
		all_sizes = np.array(model_info_martin_embeds.get("sizes"))

		# sort by product and ignore all objects that have size > 100 for their product across x, y, z
		all_sizes_prod = np.prod(model_info_martin_embeds.get("sizes"), axis=1)
		size_prod_threshold = 150.0

		all_embeds = all_embeds[all_sizes_prod < size_prod_threshold, :]
		all_sizes = all_sizes[all_sizes_prod < size_prod_threshold]
		all_jids = np.array(model_info_martin_embeds.get("jids"))[all_sizes_prod < size_prod_threshold].tolist()

		print(f"removed {all_sizes_prod[all_sizes_prod >= size_prod_threshold].shape[0]} assets with size > 100")
		print("shapes: ", all_embeds.shape, all_sizes.shape, len(all_jids))

		# cast to tensor for later
		all_embeds = torch.tensor(all_embeds)
		all_sizes = torch.tensor(all_sizes)

		# print all objects with size and desc that contain "L-shaped sofa" in their "desc".
		# we can get the "desc" via "jid" that has the same index as size
		# for i in range(all_embeds.shape[0]):
		# 	jid = all_jids[i]
		# 	asset = self.all_assets_metadata.get(jid)
		# 	if asset is not None:
		# 		desc = asset.get("summary")
		# 	else:
		# 		asset = self.all_assets_metadata_scaled.get(jid)
		# 		orig_jid = asset.get("jid")
		# 		orig_asset = self.all_assets_metadata.get(orig_jid)
		# 		desc = orig_asset.get("summary")
		# 	if "l-shaped sofa" in desc.lower():
		# 		print(f"idx: {i} — jid: {jid}")
		# 		print(f"\t desc: {desc}")
		# 		print(f"\t size: {all_sizes[i]}")
		# 		print(f"\t all_embeds[i]: {all_embeds[i]}")
		# 		print(f"\t all_sizes[i]: {all_sizes[i]}")
		# 		print("")
		# exit()

		if self.accelerator:
			# self.siglip_model = self.accelerator.prepare()
			self.siglip_model = self.siglip_model.to(accelerator.device)
			all_embeds = all_embeds.to(self.accelerator.device)
			# all_sizes = all_sizes.to(torch.float32).to(self.accelerator.device)
			all_sizes = all_sizes.to(self.accelerator.device)
		else:
			self.siglip_model = self.siglip_model.to(dvc)
			all_embeds = all_embeds.to(dvc)
			all_sizes = all_sizes.to(dvc)

		self.all_embeds_catalog = torch.nn.functional.normalize(all_embeds, p=2, dim=1)
		self.all_sizes_catalog = all_sizes
		self.all_jids_catalog = all_jids
		
		# Learnable parameters
		self.lambd = torch.tensor(lambd)
		self.sigma = torch.tensor(sigma)
		self.temp = torch.tensor(temp)

		# Fixed hyperparameters
		self.top_p = top_p
		self.top_k = top_k
		self.asset_size_threshold = asset_size_threshold
		self.do_print = do_print
		self.is_sft_training = is_sft_training

		# self.sbert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

	def get_text_embeddings(self, txts):
		# print(f"idx [{self.accelerator.process_index if self.accelerator else None}] — before siglip tokenizer")
		# print(txts)
		try:
			inputs = self.siglip_tokenizer(txts, truncation=True, padding="max_length", return_tensors="pt", return_attention_mask=True)

			if self.accelerator:
				inputs = {k: v.to(self.accelerator.device) for k, v in inputs.items()}
			else:
				inputs = {k: v.to(self.dvc) for k, v in inputs.items()}

				# print(f"idx [{self.accelerator.process_index if self.accelerator else None}] — before siglip model")
				# print(inputs)

				# print(f"Model device: {next(self.siglip_model.parameters()).device}")
				# print(f"Input device: {inputs['input_ids'].device}")

				# print(f"idx [{self.accelerator.process_index if self.accelerator else None}] — GPU Memory before forward pass:")
				# print(f"idx [{self.accelerator.process_index if self.accelerator else None}] — Allocated: {torch.cuda.memory_allocated(self.accelerator.device) / 1e9:.2f} GB")
				# print(f"idx [{self.accelerator.process_index if self.accelerator else None}] — Reserved: {torch.cuda.memory_reserved(self.accelerator.device) / 1e9:.2f} GB")
				# print(f"idx [{self.accelerator.process_index if self.accelerator else None}] — Max allocated: {torch.cuda.max_memory_allocated(self.accelerator.device) / 1e9:.2f} GB")

			# forward pass
			with torch.inference_mode():
				if self.accelerator:
					with self.accelerator.no_sync(self.siglip_model):
						outputs = self.siglip_model(**inputs)
				else:
					outputs = self.siglip_model(**inputs)
			
			embed = outputs.pooler_output

			return embed

			# print(f"idx [{self.accelerator.process_index if self.accelerator else None}] — after siglip pooled output")

		except Exception as exc:
			print(f"idx [{self.accelerator.process_index if self.accelerator else None}] — could not compute text embeddings")
			print(exc)

			return None
		
		

	def compute_text_similarity(self, text1, text2):
		embeddings = self.get_text_embeddings([text1, text2])
		similarities = torch.cosine_similarity(embeddings[0], embeddings[1], dim=0).item()

		return similarities

	def compute_semantic_similarities(self, embeds):
		embeds_norm = torch.nn.functional.normalize(embeds, p=2, dim=1)

		torch.use_deterministic_algorithms(False)
		torch.backends.cudnn.deterministic = False
		similarities = torch.matmul(self.all_embeds_catalog, embeds_norm.T)

		# similarities = similarities / 0.000000001
		# similarities = torch.pow(similarities, 0.5) / 0.1

		return similarities

	def compute_size_similarities(self, query_sizes):
		size_diffs = torch.sum(torch.square(self.all_sizes_catalog.unsqueeze(1) - query_sizes.unsqueeze(0)), dim=-1)
		similarities = torch.exp(-size_diffs / (2 * (self.sigma ** 2)))

		return similarities

	def compute_final_probabilities(self, sims_batch):
		# sims_batch: A matrix where each column corresponds to one query's similarities across all assets

		all_probs_batch = []

		# loop through each batch (each query) and apply top-k and top-p filtering
		for sims in sims_batch.T:  # Iterate over columns (each query)

			# temp scaling
			scaled_sims = sims / self.temp
			# scaled_sims = sims

			# probs_sorted = np.sort(scaled_sims.detach().numpy())[::-1][:50]
			# print(probs_sorted)
			# plt.bar(np.arange(probs_sorted.shape[0]), probs_sorted)
			# plt.show()
			# exit()
			
			# apply top-k filtering
			top_k = min(self.top_k, len(scaled_sims))
			top_k_sims, top_k_indices = torch.topk(scaled_sims, k=top_k)
			
			# apply softmax to top-k probs
			top_k_probs = torch.softmax(top_k_sims, dim=0)

			# Scatter top-k probabilities back into the full similarity tensor
			all_probs = torch.zeros_like(scaled_sims)
			all_probs.scatter_(0, top_k_indices, top_k_probs)

			# Normalize after scattering
			all_probs = all_probs / all_probs.sum()

			# Apply top-p (nucleus sampling)
			sorted_probs, sorted_indices = torch.sort(all_probs, descending=True)
			
			if self.is_sft_training:
				torch.use_deterministic_algorithms(False)
				cumulative_probs = torch.cumsum(sorted_probs, dim=0)
				torch.use_deterministic_algorithms(True)
			else:
				cumulative_probs = torch.cumsum(sorted_probs, dim=0)

			# Find which indices to remove based on top-p
			sorted_indices_to_remove = cumulative_probs > self.top_p
			sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()  # Shift by one
			sorted_indices_to_remove[0] = False  # Ensure we don't remove the first element
			indices_to_remove = sorted_indices[sorted_indices_to_remove]

			# Set probabilities at these indices to zero
			all_probs[indices_to_remove] = 0

			# Normalize again after top-p filtering
			all_probs = all_probs / all_probs.sum()

			# probs_sorted = np.sort(all_probs[all_probs > 0.0].detach().numpy())[::-1]
			# print(probs_sorted)
			# plt.bar(np.arange(probs_sorted.shape[0]), probs_sorted)
			# plt.show()
			# exit()

			# Store the final probabilities for this query
			all_probs_batch.append(all_probs)

		# Stack all probability vectors into a matrix
		return torch.stack(all_probs_batch, dim=1).T

	def forward_batch(self, query_texts, query_sizes):

		# print("before text embeddings")

		query_embeds = self.get_text_embeddings(query_texts)
		# print("after text embeds")

		semantic_sims = self.compute_semantic_similarities(query_embeds)
		# print("after semantic sims")

		query_sizes = torch.tensor(query_sizes)
		if self.accelerator:
			query_sizes = query_sizes.to(self.accelerator.device)
		else:
			query_sizes = query_sizes.to(self.dvc)

		size_sims = self.compute_size_similarities(query_sizes)

		# print("after size sims")

		weighted_sims = self.lambd * semantic_sims + (1 - self.lambd) * size_sims
		probs = self.compute_final_probabilities(weighted_sims)

		# print("after final probs")

		return probs

	def create_sampled_obj(self, obj, probs, is_greedy_sampling):

		if self.do_print:
			print(f"sampling obj with desc: {obj.get('desc')} and size {obj.get('size')}")
			n_top = min(5, self.top_k)
			idxs_top = torch.argsort(probs, descending=True)[:n_top]
			print("top probs:", torch.sort(probs, descending=True)[0].detach().cpu().numpy()[:n_top].tolist())
			jids = [ self.all_jids_catalog[idx.item()] for idx in idxs_top ]
			for idx, jid in zip(idxs_top, jids):
				asset = self.all_assets_metadata.get(jid)
				if asset == None:
					asset = self.all_assets_metadata_scaled.get(jid)
					print(jid, asset)
					orig_jid = asset.get("jid")
					orig_asset = self.all_assets_metadata.get(orig_jid)
					desc = orig_asset.get("summary")
				else:
					desc = asset.get("summary")
				print(f"")
				print(f"\t idx: [{idx}] — jid: {jid}")
				print(f"\t desc: {desc}")
				print(f"\t size: {asset.get('size')}")
			print("")

		# get jid for sampled object but skip if already set (for GT assets)
		if obj.get("jid") == None:
			if is_greedy_sampling:
				_, idx_sampled = torch.max(probs, dim=0)
			else:
				idx_sampled = torch.multinomial(probs, num_samples=1)
				if self.do_print:
					print("idx_sampled:", idx_sampled)
			jid_sampled_obj = self.all_jids_catalog[idx_sampled]
		else:
			jid_sampled_obj = obj.get("jid")

		asset = self.all_assets_metadata.get(jid_sampled_obj)
		if asset == None:
			asset = self.all_assets_metadata_scaled.get(jid_sampled_obj)
			size_sampled_obj = asset.get("size")
			orig_jid = asset.get("jid")
			orig_asset = self.all_assets_metadata.get(orig_jid)
			desc_sampled_obj = orig_asset.get("summary")
		else:
			desc_sampled_obj = asset.get("summary")
			size_sampled_obj = asset.get("size")

		new_obj = copy.deepcopy(obj)
		new_obj.update({
			"sampled_asset_jid": jid_sampled_obj,
			"sampled_asset_desc": desc_sampled_obj,
			"sampled_asset_size": size_sampled_obj,
			"uuid": str(uuid.uuid4())
		})

		if self.do_print:
			print(obj)
			print("\n")

		return new_obj

	def sample_all_assets(self, scene, batch_size=64, is_greedy_sampling=True):

		if self.do_print: 
			print(f"sampling full scene... (# of objects: {len(scene.get('objects', []))})")

		sampled_scene = copy.deepcopy(scene)
		sampled_scene["objects"] = []
		desc_size_map = {}

		descriptions = [obj.get("desc") for obj in scene.get("objects", [])]
		sizes = [obj.get("size", []) for obj in scene.get("objects", [])]
		
		for batch_start in range(0, len(descriptions), batch_size):
			batch_end = min(batch_start + batch_size, len(descriptions))

			batch_descriptions = descriptions[batch_start:batch_end]
			batch_sizes = sizes[batch_start:batch_end]

			batch_probs = self.forward_batch(batch_descriptions, batch_sizes)

			for i, obj in enumerate(scene.get("objects", [])[batch_start:batch_end]):
				desc = obj.get("desc")
				size = obj.get("size", [])

				if desc in desc_size_map:
					# Try to find existing sampled assets with the same description and size within the threshold
					matching_obj = None
					for sampled_obj in desc_size_map[desc]:
						if self.calculate_size_difference(size, sampled_obj["size"]) <= self.asset_size_threshold:
							matching_obj = sampled_obj
							break
					if matching_obj:
						new_obj = copy.deepcopy(obj)
						new_obj.update({
							"sampled_asset_jid": matching_obj["sampled_asset_jid"],
							"sampled_asset_desc": matching_obj["sampled_asset_desc"],
							"sampled_asset_size": matching_obj["sampled_asset_size"],
							"uuid": str(uuid.uuid4())
						})
					else:
						new_obj = self.create_sampled_obj(obj, batch_probs[i], is_greedy_sampling)
						desc_size_map[desc].append(new_obj)
				else:
					new_obj = self.create_sampled_obj(obj, batch_probs[i], is_greedy_sampling)
					desc_size_map[desc] = [new_obj]

				sampled_scene["objects"].append(new_obj)

		return sampled_scene
	
	# sample only last object in the list
	def sample_last_asset(self, scene, is_greedy_sampling=True):

		if self.do_print:
			print(f"sampling last object in scene...")

		sampled_scene = copy.deepcopy(scene)
		sampled_scene["objects"] = scene.get("objects", [])[:-1]

		if len(scene.get("objects", [])) > 0:
			last_obj = scene.get("objects")[-1]
			desc = last_obj.get("desc")
			size = last_obj.get("size", [])

			probs = self.forward_batch([desc], [size])

			new_obj = self.create_sampled_obj(last_obj, probs[0], is_greedy_sampling)
			sampled_scene["objects"].append(new_obj)

		return sampled_scene

	@staticmethod
	def calculate_size_difference(size1, size2):
		return np.linalg.norm(np.array(size1) - np.array(size2))

# **********************************************************************************************************

# def train(model, optimizer, queries, sizes, num_epochs, batch_size):

# 	for epoch in range(num_epochs):
# 		for i in range(0, len(queries), batch_size):
# 			batch_queries = queries[i:i+batch_size]
# 			batch_sizes = sizes[i:i+batch_size]

# 			optimizer.zero_grad()
# 			total_loss = 0

# 			for query, size in zip(batch_queries, batch_sizes):
# 				probs = model(query, size)
# 				samples = model.sample(probs)

# 				# Here, you would use GPT-4V to rate the samples
# 				ratings = rate_samples_with_gpt4v(query, size, samples)

# 				pos_samples = samples[ratings > 8]
# 				neg_samples = samples[ratings < 3]

# 				if len(pos_samples) > 0 and len(neg_samples) > 0:
# 					pos_probs = probs[pos_samples]
# 					neg_probs = probs[neg_samples]

# 					# Contrastive loss
# 					loss = -torch.log(pos_probs.sum()) - torch.log(1 - neg_probs.sum())

# 					total_loss += loss

# 			if total_loss > 0:
# 				total_loss.backward()
# 				optimizer.step()

# 		print(f"Epoch {epoch+1}, Loss: {total_loss.item()}, "
# 			  f"Top-k: {model.top_k_param.item()}, Top-p: {model.top_p_param.item()}")

# **********************************************************************************************************
# train

# Initialize the model
# model = AssetRetrievalModule(siglip_model, siglip_tokenizer, all_embeds_catalog, all_sizes_catalog, all_jids_catalog)

# Create an optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
# train(model, optimizer, queries, sizes, num_epochs=10, batch_size=32)

# **********************************************************************************************************
# test

if __name__ == "__main__":

	load_dotenv(".env.local")

	sampling_engine = AssetRetrievalModule(lambd=0.5, sigma=0.05, temp=0.2, top_p=0.95, top_k=20, asset_size_threshold=0.5, rand_seed=1234, do_print=True)

	# pth_src = "/Volumes/apollo11/data/3D-FRONT-martin-rooms-train-v1/946dac58-6180-434b-a23e-ecebc7704411-a43d4a4d-4c6a-4158-a26c-f6d2f019e1e7.json"
	# scene = json.load(open(pth_src))

	# scene = json.loads('{"objects":[{"desc":"Modern minimalist white desk with metal frame, wooden rectangular top, and open leg design.","size":[1.2,0.75,0.6],"pos":[0.17,0.0,0.71],"rot":[0,0.70711,0,0.70711]},{"desc":"Modern ergonomic chair with a mesh backrest, padded seat, foldable arms, and metal frame in black, gray, and white colors.","size":[0.64,1.02,0.68],"pos":[0.78,0.0,0.97],"rot":[0,-0.70711,0,0.70711]},{"desc":"Modern minimalist lounge chair with black fabric cushions, natural wood armrests, and a sleek metallic frame.","size":[0.92,0.93,0.92],"pos":[-0.78,0.0,-0.31],"rot":[0,0.38268,0,0.92388]},{"desc":"This contemporary-traditional rectangular bookcase in white wood features symmetrical open shelves, bottom cabinets with metal handles, making it a versatile and elegant storage solution.","size":[2.0,2.51,0.4],"pos":[0.24,0.0,-0.97],"rot":[0,0,0,1]}]}')
	
	# scene = json.loads('{"bounds_top": [[-1.9, 2.7, 1.9], [1.9, 2.7, 1.9], [1.9, 2.7, -1.9], [-1.9, 2.7, -1.9]], "bounds_bottom": [[-1.9, 0.0, 1.9], [1.9, 0.0, 1.9], [1.9, 0.0, -1.9], [-1.9, 0.0, -1.9]], "room_type": "bedroom", "objects": [{"desc": "Minimalist modern brown wooden open back bookcase with clean lines, six shelves per unit, and a paired unit structure.", "size": [2.09, 2.0, 0.3], "pos": [-1.72, 0.0, -0.82], "rot": [0, 0.70711, 0, 0.70711]}]}')
	# scene = json.loads('{"bounds_top": [[2.69, 2.7, 0.01], [0.01, 2.7, -2.69], [-2.69, 2.7, -0.01], [-0.01, 2.7, 2.69]], "bounds_bottom": [[2.69, 0.0, 0.01], [0.01, 0.0, -2.69], [-2.69, 0.0, -0.01], [-0.01, 0.0, 2.69]], "room_type": "bedroom", "objects": [{"desc": "Minimalist modern brown wooden open back bookcase with clean lines, six shelves per unit, and a paired unit structure.", "size": [2.09, 2.0, 0.3], "pos": [0.63, 0.0, 1.8], "rot": [0.0, 0.924431453527687, 0.0, -0.3813482499352634]}]}')
	# scene = json.loads('{"bounds_top": [[-1.95, 2.6, 3.1], [1.95, 2.6, 3.1], [1.95, 2.6, -1.1], [-0.65, 2.6, -1.1], [-0.65, 2.6, -3.1], [-1.95, 2.6, -3.1]], "bounds_bottom": [[-1.95, 0.0, 3.1], [1.95, 0.0, 3.1], [1.95, 0.0, -1.1], [-0.65, 0.0, -1.1], [-0.65, 0.0, -3.1], [-1.95, 0.0, -3.1]], "room_type": "bedroom", "objects": [{"desc": "Modern minimalist nightstand with black wood frame, white drawers, and sleek metal legs.", "size": [0.63, 0.51, 0.39], "pos": [1.76, 0.0, -0.01], "rot": [0, -0.70711, 0, 0.70711]}, {"desc": "Modern minimalist nightstand with black wood frame, white drawers, and sleek metal legs.", "size": [0.63, 0.51, 0.39], "pos": [1.84, 0.0, 2.7], "rot": [0, -0.70711, 0, 0.70711]}, {"desc": "Modern minimalist king-size bed with dark brown fabric upholstery, low-profile wooden frame, and sleek design.", "size": [2.3, 0.78, 2.42], "pos": [0.74, 0.0, 1.28], "rot": [0, -0.70711, 0, 0.70711]}, {"desc": "A modern minimalist dark gray wardrobe with sliding mirror doors, shelves, and a hanging rod.", "size": [2.54, 2.43, 0.78], "pos": [0.67, 0.0, -0.72], "rot": [0, 0, 0, 1]}]}')
	# scene = json.loads('{"bounds_top": [[2.6869945511849864, 2.7, 0.007764142076983778], [0.007764142076983852, 2.7, -2.6869945511849864], [-2.6869945511849864, 2.7, -0.007764142076983778], [-0.007764142076983852, 2.7, 2.6869945511849864]], "bounds_bottom": [[2.6869945511849864, 0.0, 0.007764142076983778], [0.007764142076983852, 0.0, -2.6869945511849864], [-2.6869945511849864, 0.0, -0.007764142076983778], [-0.007764142076983852, 0.0, 2.6869945511849864]], "room_type": "bedroom", "objects": [{"desc": "Minimalist modern brown wooden open back bookcase with clean lines, six shelves per unit, and a paired unit structure.", "size": [2.09, 2.0, 0.3], "pos": [0.6312037303134077, 0.0, 1.7978826020734608], "rot": [0, 0.70711, 0, 0.70711]}]}')

	# scene = json.loads('{"bounds_top": [[1.96, 2.6, 0.98], [1.61, 2.6, -1.49], [-1.96, 2.6, -0.98], [-1.61, 2.6, 1.49]], "bounds_bottom": [[1.96, 0.0, 0.98], [1.61, 0.0, -1.49], [-1.96, 0.0, -0.98], [-1.61, 0.0, 1.49]], "room_type": "bedroom", "objects": [{"desc": "Contemporary minimalist desk featuring brown and white wood composite materials with open cubbies, closed shelves, double-door cabinets, and sleek handles.", "size": [1.4, 1.86, 0.61], "pos": [-0.84, 0.0, 1.11], "rot": [0.0, 0.9974822068295638, 0.0, -0.07091718450716644]}, {"desc": "A modern, avant-garde pendant lamp with a sculptural gold frame, featuring candle-like glass tubes creating a contemporary, linear aesthetic.", "size": [1.68, 0.69, 0.32], "pos": [-0.13, 1.96, -0.14], "rot": [0.0, 0.6551804104944319, 0.0, -0.7554724546297819]}, {"desc": "A playful animal-shaped desk made of vibrant colors and whimsical patterns, adding a fun touch to the room.", "size": [1.2, 1.2, 0.5], "pos": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0, 1.0]}]}')

	# scene = json.loads('{"bounds_top": [[-1.7, 2.6, 2.1], [1.7, 2.6, 2.1], [1.7, 2.6, -2.1], [-1.7, 2.6, -2.1]], "bounds_bottom": [[-1.7, 0.0, 2.1], [1.7, 0.0, 2.1], [1.7, 0.0, -2.1], [-1.7, 0.0, -2.1]], "room_type": "bedroom", "objects": [{"desc": "A minimalist modern ceramic vase with sleek black branches, perfect for contemporary aesthetics.", "size": [0.13, 0.49, 0.31], "pos": [1.41, 0.0, 1.28], "rot": [0, -0.70711, 0, 0.70711]}, {"desc": "A modern minimalist gray wooden nightstand with three drawers and angled metal legs, boasting sleek and clean lines.", "size": [0.51, 0.56, 0.5], "pos": [-1.44, 0.0, 1.05], "rot": [0, 0.70711, 0, 0.70711]}, {"desc": "Modern bed frame with dark wood and padded dark gray leather headboard, blending simplicity and mid-century elements.", "size": [2.04, 1.08, 2.37], "pos": [-0.51, 0.0, -0.28], "rot": [0, 0.70711, 0, 0.70711]}, {"desc": "A sleek, black, contemporary wardrobe with clean minimalist lines, four doors, and subtle metal handles.", "size": [2.0, 2.36, 0.66], "pos": [-0.69, 0.0, 1.72], "rot": [0, 1, 0, 0]}, {"desc": "A modern minimalist gray wooden nightstand with three drawers and angled metal legs, boasting sleek and clean lines.", "size": [0.51, 0.56, 0.5], "pos": [-1.46, 0.0, -1.62], "rot": [0, 0.70711, 0, 0.70711]}]}')

	# scene = json.loads('{"bounds_top": [[-1.25, 2.8, 1.85], [1.25, 2.8, 1.85], [1.25, 2.8, -1.85], [-1.25, 2.8, -1.85]], "bounds_bottom": [[-1.25, 0.0, 1.85], [1.25, 0.0, 1.85], [1.25, 0.0, -1.85], [-1.25, 0.0, -1.85]], "room_type": "bedroom", "objects": [{"desc": "A contemporary minimalist pendant lamp featuring a linear metal design with gray and gold finish, characterized by suspended circular accents.", "size": [1.09, 0.89, 0.05], "pos": [-0.03, 1.98, -0.28], "rot": [0, 0, 0, 1]}, {"desc": "An antique-inspired light gray wooden nightstand with two drawers, an open shelf, ornate accents, and brass knobs.", "size": [0.3, 0.44, 0.24], "pos": [0.99, 0.0, -1.37], "rot": [0, -0.70711, 0, 0.70711]}, {"desc": "Classic grey wooden wardrobe with 4 louvered doors, crown molding, and base plinth.", "size": [2.04, 2.25, 0.65], "pos": [0.22, 0.0, 1.52], "rot": [0, -1, 0, 0]}]}')
	# scene = json.loads('{"bounds_top": [[-1.85, 2.8, -1.25], [-1.85, 2.8, 1.25], [1.85, 2.8, 1.25], [1.85, 2.8, -1.25]], "bounds_bottom": [[-1.85, 0.0, -1.25], [-1.85, 0.0, 1.25], [1.85, 0.0, 1.25], [1.85, 0.0, -1.25]], "room_type": "bedroom", "objects": [{"desc": "A contemporary minimalist pendant lamp featuring a linear metal design with gray and gold finish, characterized by suspended circular accents.", "size": [1.09, 0.89, 0.05], "pos": [0.28, 1.98, -0.03], "rot": [0.0, 0.70711, 0.0, -0.70711]}, {"desc": "An antique-inspired light gray wooden nightstand with two drawers, an open shelf, ornate accents, and brass knobs.", "size": [0.3, 0.44, 0.24], "pos": [1.37, 0.0, 0.99], "rot": [0.0, 1.0, 0.0, 0.0]}, {"desc": "Classic grey wooden wardrobe with 4 louvered doors, crown molding, and base plinth.", "size": [2.04, 2.25, 0.65], "pos": [-1.52, 0.0, 0.22], "rot": [0.0, 0.70711, 0.0, 0.70711]}]}')
	# scene = json.loads('{"bounds_top": [[1.85, 2.8, -1.25], [1.85, 2.8, 1.25], [-1.85, 2.8, 1.25], [-1.85, 2.8, -1.25]], "bounds_bottom": [[1.85, 0.0, -1.25], [1.85, 0.0, 1.25], [-1.85, 0.0, 1.25], [-1.85, 0.0, -1.25]], "room_type": "bedroom", "objects": [{"desc": "A contemporary minimalist pendant lamp featuring a linear metal design with gray and gold finish, characterized by suspended circular accents.", "size": [1.09, 0.89, 0.05], "pos": [-0.28, 1.98, -0.03], "rot": [0.0, 0.70711, 0.0, -0.70711]}, {"desc": "An antique-inspired light gray wooden nightstand with two drawers, an open shelf, ornate accents, and brass knobs.", "size": [0.3, 0.44, 0.24], "pos": [-1.37, 0.0, 0.99], "rot": [0.0, 1.0, 0.0, 0.0]}, {"desc": "Classic grey wooden wardrobe with 4 louvered doors, crown molding, and base plinth.", "size": [2.04, 2.25, 0.65], "pos": [1.52, 0.0, 0.22], "rot": [0.0, 0.70711, 0.0, 0.70711]}]}')
	
	# scene = json.loads('{"bounds_top": [[1.55, 2.6, 2.5], [1.55, 2.6, -2.5], [-1.55, 2.6, -2.5], [-1.55, 2.6, 2.5]], "bounds_bottom": [[1.55, 0.0, 2.5], [1.55, 0.0, -2.5], [-1.55, 0.0, -2.5], [-1.55, 0.0, 2.5]], "room_type": "bedroom", "objects": [{"desc": "Modern king-size bed with a quilted high-back headboard, side panels, and plush cushions accented by geometric pillows.", "size": [2.0, 0.97, 2.08], "pos": [-0.49, 0.0, -0.84], "rot": [0.0, -0.70711, 0.0, -0.70711]}, {"desc": "Modern minimalist nightstand with gray upholstery, single drawer, and distinctive curved wood legs.", "size": [0.44, 0.44, 0.35], "pos": [-1.32, 0.0, -2.04], "rot": [0.0, 0.70711, 0.0, 0.70711]}, {"desc": "Modern minimalist nightstand with gray upholstery, single drawer, and distinctive curved wood legs.", "size": [0.44, 0.44, 0.35], "pos": [-1.33, 0.0, 0.35], "rot": [0.0, 0.70711, 0.0, 0.70711]}]}')
	# scene = json.loads('{"bounds_top": [[1.55, 2.6, -2.5], [1.55, 2.6, 2.5], [-1.55, 2.6, 2.5], [-1.55, 2.6, -2.5]], "bounds_bottom": [[1.55, 0.0, -2.5], [1.55, 0.0, 2.5], [-1.55, 0.0, 2.5], [-1.55, 0.0, -2.5]], "room_type": "bedroom", "objects": [{"desc": "Modern king-size bed with a quilted high-back headboard, side panels, and plush cushions accented by geometric pillows.", "size": [2.0, 0.97, 2.08], "pos": [-0.49, 0.0, 0.84], "rot": [0.0, 0.70711, 0.0, -0.70711]}, {"desc": "Modern minimalist nightstand with gray upholstery, single drawer, and distinctive curved wood legs.", "size": [0.44, 0.44, 0.35], "pos": [-1.32, 0.0, 2.04], "rot": [0.0, 0.70711, 0.0, -0.70711]}, {"desc": "Modern minimalist nightstand with gray upholstery, single drawer, and distinctive curved wood legs.", "size": [0.44, 0.44, 0.35], "pos": [-1.33, 0.0, -0.35], "rot": [0.0, 0.70711, 0.0, -0.70711]}]}')
	
	# test_local_rotations()
	# exit()

	scene = json.loads('{"bounds_top": [[-1.35, 2.6, 1.45], [0.15, 2.6, 1.45], [0.15, 2.6, 2.15], [1.35, 2.6, 2.15], [1.35, 2.6, -2.15], [-1.35, 2.6, -2.15]], "bounds_bottom": [[-1.35, 0.0, 1.45], [0.15, 0.0, 1.45], [0.15, 0.0, 2.15], [1.35, 0.0, 2.15], [1.35, 0.0, -2.15], [-1.35, 0.0, -2.15]], "room_type": "bedroom", "objects": [{"desc": "Mid-Century Modern nightstand with a wood construction, featuring a rectangular shape, single drawer, tapered legs, and a distinctive cut-out handle.", "size": [0.55, 0.45, 0.43], "pos": [1.1, 0.0, 0.95], "rot": [0, -0.70711, 0, 0.70711]}, {"desc": "Mid-Century Modern nightstand with a wood construction, featuring a rectangular shape, single drawer, tapered legs, and a distinctive cut-out handle.", "size": [0.55, 0.45, 0.43], "pos": [1.23, 0.0, -1.42], "rot": [0, -0.70711, 0, 0.70711]}]}')
	# scene = do_random_augm_on_sgs(scene)

	print(scene)

	# sampled_scene = scene

	# sampled_scene = scene

	# scene = json.loads('{"bounds_top": [[-1.7, 2.6, 2.1], [1.7, 2.6, 2.1], [1.7, 2.6, -2.1], [-1.7, 2.6, -2.1]], "bounds_bottom": [[-1.7, 0.0, 2.1], [1.7, 0.0, 2.1], [1.7, 0.0, -2.1], [-1.7, 0.0, -2.1]], "room_type": "bedroom", "objects": [{"desc": "A minimalist modern ceramic vase with sleek black branches, perfect for contemporary aesthetics.", "size": [0.13, 0.49, 0.31], "pos": [1.41, 0.0, 1.28], "rot": [0, -0.70711, 0, 0.70711]}, {"desc": "A modern minimalist gray wooden nightstand with three drawers and angled metal legs, boasting sleek and clean lines.", "size": [0.51, 0.56, 0.5], "pos": [-1.44, 0.0, 1.05], "rot": [0, 0.70711, 0, 0.70711]}, {"desc": "Modern bed frame with dark wood and padded dark gray leather headboard, blending simplicity and mid-century elements.", "size": [2.04, 1.08, 2.37], "pos": [-0.51, 0.0, -0.28], "rot": [0, 0.70711, 0, 0.70711]}, {"desc": "A sleek, black, contemporary wardrobe with clean minimalist lines, four doors, and subtle metal handles.", "size": [2.0, 2.36, 0.66], "pos": [-0.69, 0.0, 1.72], "rot": [0, 1, 0, 0]}, {"desc": "A modern minimalist gray wooden nightstand with three drawers and angled metal legs, boasting sleek and clean lines.", "size": [0.51, 0.56, 0.5], "pos": [-1.46, 0.0, -1.62], "rot": [0, 0.70711, 0, 0.70711]}]}')
	sampled_scene = sampling_engine.sample_all_assets(scene, is_greedy_sampling=True)

	print(sampled_scene)

	# scene = json.loads('{"bounds_top": [[1.55, 2.6, -2.5], [1.55, 2.6, 2.5], [-1.55, 2.6, 2.5], [-1.55, 2.6, -2.5]], "bounds_bottom": [[1.55, 0.0, -2.5], [1.55, 0.0, 2.5], [-1.55, 0.0, 2.5], [-1.55, 0.0, -2.5]], "room_type": "bedroom", "objects": [{"desc": "Modern king-size bed with a quilted high-back headboard, side panels, and plush cushions accented by geometric pillows.", "size": [2.0, 0.97, 2.08], "pos": [-0.49, 0.0, 0.84], "rot": [0.0, -0.70711, 0.0, -0.70711]}, {"desc": "Modern minimalist nightstand with gray upholstery, single drawer, and distinctive curved wood legs.", "size": [0.44, 0.44, 0.35], "pos": [-1.32, 0.0, 2.04], "rot": [0.0, 0.70711, 0.0, 0.70711]}, {"desc": "Modern minimalist nightstand with gray upholstery, single drawer, and distinctive curved wood legs.", "size": [0.44, 0.44, 0.35], "pos": [-1.33, 0.0, -0.35], "rot": [0.0, 0.70711, 0.0, 0.70711]}]}')

	# output = "{'bounds_top': [[1.96, 2.6, 0.98], [1.61, 2.6, -1.49], [-1.96, 2.6, -0.98], [-1.61, 2.6, 1.49]], 'bounds_bottom': [[1.96, 0.0, 0.98], [1.61, 0.0, -1.49], [-1.96, 0.0, -0.98], [-1.61, 0.0, 1.49]], 'room_type': 'bedroom', 'objects': [{'desc': 'Contemporary minimalist desk featuring brown and white wood composite materials with open cubbies, closed shelves, double-door cabinets, and sleek handles.', 'size': [1.4, 1.86, 0.61], 'pos': [-0.84, 0.0, 1.11], 'rot': [0.0, 0.9974822068295638, 0.0, -0.07091718450716644], 'sampled_asset_jid': 'fec2a457-251b-40ad-94b8-ee4760e78153', 'sampled_asset_desc': 'Contemporary minimalist desk featuring brown and white wood composite materials with open cubbies, closed shelves, double-door cabinets, and sleek handles.', 'sampled_asset_size': [1.399999976158142, 1.8640500122123846, 0.6095100939273834], 'uuid': '2e4fd652-5110-43b0-82aa-4d68c2b4cc4f'}, {'desc': 'A modern, avant-garde pendant lamp with a sculptural gold frame, featuring candle-like glass tubes creating a contemporary, linear aesthetic.', 'size': [1.68, 0.69, 0.32], 'pos': [-0.13, 1.96, -0.14], 'rot': [0.0, 0.6551804104944319, 0.0, -0.7554724546297819], 'sampled_asset_jid': 'da2f4cae-4c21-4671-aaaf-2eb1f5424f16', 'sampled_asset_desc': 'A modern, avant-garde pendant lamp with a sculptural gold frame, featuring candle-like glass tubes creating a contemporary, linear aesthetic.', 'sampled_asset_size': [1.6789209842681885, 0.692858770955354, 0.318820983171463], 'uuid': 'c3dc3408-ef23-484a-bd03-29cf2fadf44b'}, {'desc': 'A playful animal-shaped desk made of vibrant colors and whimsical patterns, adding a fun touch to the room.', 'size': [1.2, 1.2, 0.5], 'pos': [0.0, 0.0, 0.0], 'rot': [0.0, 0.0, 0.0, 1.0], 'sampled_asset_jid': 'c56b454e-5c55-4dcb-bccd-3c67357bc9ec-(1.2)-(1.0)-(1.09)', 'sampled_asset_desc': 'Minimalist beige modern dressing table with hidden storage, rectangular mirror, and unique curved leg design.', 'sampled_asset_size': [1.2, 1.22, 0.51], 'uuid': '48104a85-0162-45c9-972b-1055747dee5b'}]}"

	# 0.48

	# 0.4609

	# score = sampling_engine.compute_text_similarity("modern king-size bed", "A modern minimalist king-size bed with a padded headboard, solid wood frame, and decorative blue, yellow, and white pillows")
	# score = sampling_engine.compute_text_similarity("Modern minimalist black wood nightstand with drawers and metal legs", "Modern minimalist nightstand with black wood frame, white drawers, and sleek metal legs.")
	# print(score)
	
	# score = sampling_engine.compute_text_similarity("blue metal nightstand", "Modern minimalist nightstand with black wood frame, white drawers, and sleek metal legs.")
	# print(score)

	# print("==")

	# score = sampling_engine.compute_text_similarity("modern dark gray wardrobe", "A modern minimalist dark gray wardrobe with sliding mirror doors, shelves, and a hanging rod.")
	# print(score)
	
	# score = sampling_engine.compute_text_similarity("blue wardrobe", "A modern minimalist dark gray wardrobe with sliding mirror doors, shelves, and a hanging rod.")
	# print(score)

	# exit()
	#sampled_scene = sampling_engine.sample_all_assets(scene, is_greedy_sampling=False)
	#print(sampled_scene)

	# Save the sampled scene
	pth_tgt = "/Volumes/apollo11/data/sampled/example7.json"
	with open(pth_tgt, "w") as fp:
		json.dump(sampled_scene, fp, indent=4)

	# # temp = 0.0000000001
	# # temp = 0.6

	# sigma = 0.1
	# lambd = 0.3
	# temp = 10.0
	# size_threshold = 0.5

	# # scene = json.loads('{"objects": [{"desc": "Modern minimalist white desk with metal frame, wooden rectangular top, and open leg design.", "size": [1.2, 0.75, 0.6], "pos": [0.17, 0.0, 0.71], "rot": [0, 0.70711, 0, 0.70711]}, {"desc": "Modern ergonomic chair with a mesh backrest, padded seat, foldable arms, and metal frame in black, gray, and white colors.", "size": [0.64, 1.02, 0.68], "pos": [0.78, 0.0, 0.97], "rot": [0, -0.70711, 0, 0.70711]}, {"desc": "Modern minimalist lounge chair with black fabric cushions, natural wood armrests, and a sleek metallic frame.", "size": [0.92, 0.93, 0.92], "pos": [-0.78, 0.0, -0.31], "rot": [0, 0.38268, 0, 0.92388]}, {"desc": "Contemporary-traditional rectangular bookcase with a wooden frame, multiple shelves, and a mix of glass and wood materials.", "size": [1.38, 0.86, 1.22], "pos": [0.0, 0.0, 0.0], "rot": [0, 0, 0, 1]}]}')
	# # scene = json.loads('{"objects":[{"desc":"Modern minimalist white desk with metal frame, wooden rectangular top, and open leg design.","size":[1.2,0.75,0.6],"pos":[0.17,0.0,0.71],"rot":[0,0.70711,0,0.70711]},{"desc":"Modern ergonomic chair with a mesh backrest, padded seat, foldable arms, and metal frame in black, gray, and white colors.","size":[0.64,1.02,0.68],"pos":[0.78,0.0,0.97],"rot":[0,-0.70711,0,0.70711]},{"desc":"Modern minimalist lounge chair with black fabric cushions, natural wood armrests, and a sleek metallic frame.","size":[0.92,0.93,0.92],"pos":[-0.78,0.0,-0.31],"rot":[0,0.38268,0,0.92388]},{"desc":"Traditional wooden bookcase with four shelves, made of solid white wood, and a simple design.","size":[1.35,2.33,0.75],"pos":[0.78,0.0,0.47],"rot":[0,0,0,1]}]}')
	# # scene = json.loads('{"objects":[{"desc":"Modern minimalist white desk with metal frame, wooden rectangular top, and open leg design.","size":[1.2,0.75,0.6],"pos":[0.17,0.0,0.71],"rot":[0,0.70711,0,0.70711]},{"desc":"Modern ergonomic chair with a mesh backrest, padded seat, foldable arms, and metal frame in black, gray, and white colors.","size":[0.64,1.02,0.68],"pos":[0.78,0.0,0.97],"rot":[0,-0.70711,0,0.70711]},{"desc":"Modern minimalist lounge chair with black fabric cushions, natural wood armrests, and a sleek metallic frame.","size":[0.92,0.93,0.92],"pos":[-0.78,0.0,-0.31],"rot":[0,0.38268,0,0.92388]},{"desc":"This modern minimalist white wood bookcase has rectangular open shelves, metal frame, and wooden vertical panels.","size":[2.0,2.51,0.4],"pos":[0.24,0.0,-0.97],"rot":[0,0,0,1]}]}')

	# descs_idxs = {}
	# idx = 0

	# sampled_scene = process_scene(scene, lambd, sigma, temp, size_threshold)

	# with open(pth_tgt, "w") as write_file:
	# 	json.dump(sampled_scene, write_file, indent=4)