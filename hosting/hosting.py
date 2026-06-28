
from huggingface_hub import HfApi, login
import os

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable not found.")

login(token=HF_TOKEN)

api = HfApi(token=HF_TOKEN)

api.upload_folder(
    folder_path="deployment",
    repo_id="RahulKakde/Portfolio",
    repo_type="space",
)

print("Deployment uploaded successfully to Hugging Face Space.")
