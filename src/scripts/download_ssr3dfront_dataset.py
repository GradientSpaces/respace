import requests
import zipfile

# Download raw dataset
url = "https://huggingface.co/datasets/gradient-spaces/SSR-3DFRONT/resolve/main/raw_scenes.zip"
response = requests.get(url)
with open("ssr_3dfront_raw.zip", "wb") as f:
	f.write(response.content)

# Extract
with zipfile.ZipFile("ssr_3dfront_raw.zip", 'r') as zip_ref:
	zip_ref.extractall("dataset-ssr3dfront/")