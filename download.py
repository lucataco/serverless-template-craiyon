# In this file, we define download_model
# It runs during container build time to get model weights built into the container

import torch
from min_dalle import MinDalle

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    
    # Craiyon (dall-e mini)
    model = MinDalle(
        dtype=getattr(torch, "float32"),
        is_mega=True, 
        is_reusable=True
    )


if __name__ == "__main__":
    download_model()