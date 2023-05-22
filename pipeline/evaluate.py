import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch

def predict(model, image):
    prompts = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        text = model(image)
        prompts.append(text)
    return prompts

def get_score(true, predict):
    return cosine_similarity(true, predict).mean()
