from huggingface_hub import HfApi
import os

token = userdata.get("HF_TOKEN")
api = HfApi(token=token)
api.upload_folder(
    folder_path="Portfolio/deployment",     # the local folder containing your files
    repo_id="RahulKakde/Portfolio",          # the target repo
    repo_type="space",                      # dataset, model, or space
)
