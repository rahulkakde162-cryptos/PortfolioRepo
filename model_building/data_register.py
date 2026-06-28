
from huggingface_hub import HfApi, create_repo, login
from huggingface_hub.utils import RepositoryNotFoundError
import os

repo_id = "RahulKakde/PortfolioDataset"
repo_type = "dataset"

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable not found.")

login(token=HF_TOKEN)

api = HfApi(token=HF_TOKEN)

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable not found.")

# Login
login(token=HF_TOKEN)

# Create API object
api = HfApi(token=HF_TOKEN)

# Check whether the dataset repository exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repository '{repo_id}' already exists.")

except RepositoryNotFoundError:
    print(f"Dataset repository '{repo_id}' not found. Creating...")

    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False,
        token=HF_TOKEN,
    )

# Upload dataset
api.upload_folder(
    folder_path="data",
    repo_id=repo_id,
    repo_type=repo_type,
)

print("Dataset uploaded successfully.")
