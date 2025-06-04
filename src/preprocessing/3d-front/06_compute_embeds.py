import torch
import numpy as np
import pickle
import json
import pdb
from transformers import AutoTokenizer, SiglipTextModel
from tqdm import tqdm
from dotenv import load_dotenv
import os

def get_batch_embeds(texts, batch_size=32, device='cuda'):
	embeds = []
	siglip_model.to(device)
	siglip_model.eval()
	
	for i in tqdm(range(0, len(texts), batch_size)):
		batch_texts = texts[i:i + batch_size]
		inputs = siglip_tokenizer(batch_texts, padding="max_length", return_tensors="pt", truncation=True)

		inputs = {k: v.to(device) for k, v in inputs.items()}

		with torch.no_grad():
			outputs = siglip_model(**inputs)
			pooled_output = outputs.pooler_output
			embeds.append(pooled_output.cpu().numpy())

	return np.vstack(embeds)

# **********************************************************************************************************

load_dotenv(".env")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

all_assets_metadata = json.load(open(os.getenv("PTH_ASSETS_METADATA")))
all_assets_metadata_scaled = json.load(open(os.getenv("PTH_ASSETS_METADATA_SCALED")))

siglip_model = SiglipTextModel.from_pretrained("google/siglip-so400m-patch14-384")
siglip_tokenizer = AutoTokenizer.from_pretrained("google/siglip-so400m-patch14-384")

all_jids = []
all_sizes = []
all_summaries = []

print("len:", len(all_assets_metadata.items()))
print("len:", len(all_assets_metadata_scaled.items()))

for key, val in tqdm(all_assets_metadata.items()):
	all_jids.append(key)
	all_sizes.append([round(elem, 2) for elem in val.get("size")])
	all_summaries.append(val.get("summary"))

all_embeds = get_batch_embeds(all_summaries)

for key, val in tqdm(all_assets_metadata_scaled.items()):
	idx_orig_asset = all_jids.index(val.get("jid"))
	embed_orig_asset = all_embeds[idx_orig_asset]

	all_jids.append(key)
	all_sizes.append([round(elem, 2) for elem in val.get("size")])
	all_embeds = np.vstack((all_embeds, embed_orig_asset))

all_sizes = np.array(all_sizes)

model_info_martin_embeds = {
	"jids": all_jids,
	"sizes": all_sizes,
	"embeds": all_embeds
}

with open(os.getenv("PTH_ASSETS_EMBED"), 'wb') as fp:
	pickle.dump(model_info_martin_embeds, fp)