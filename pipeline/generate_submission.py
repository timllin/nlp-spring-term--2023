import os
import numpy as np
import Path
from model.config import config
import pandas as pd

def create_imgId_eId(str_path):
    comp_path = Path(str_path)
    images = os.listdir(comp_path / 'images')
    imgIds = [i.split('.')[0] for i in images]
    eIds = list(range(config['embedding_length']))

    imgId_eId = [
        '_'.join(map(str, i)) for i in zip(
            np.repeat(imgIds, config['embedding_length']),
            np.tile(range(config['embedding_length']), len(imgIds)))]
    return imgId_eId



def to_csv(imgId_eId, prompt_embeddings):
    df = pd.DataFrame(ndex=imgId_eId, data=prompt_embeddings, columns=['val']).rename_axis('imgId_eId').reset_index()
    df.to_csv('submission.csv', index=False)
