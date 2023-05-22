import torch
from model.config import config
from sentence_transformers import SentenceTransformer
from PIL import Image
import open_clip

class ImageToPromtModel(torch.nn.Module):
    def __init__(self, config):
        super(ImageToPromtModel, self).__init__()
        self.config = config
        self.device = config['device']
        self.clip_model = config['clip_model']
        self.clip_model_transform = config['clip_model_transform']
        self.sentence_transfromer = config['sentence_transformer_model']

        self.model, _, self.transform = open_clip.create_model_and_transforms(model_name=config['clip_model'],
                                                                         pretrained=config['clip_model_transform'])
        self.st_model = SentenceTransformer(config['sentence_transformer_model'])

    def extract(self, x):
        x = self.transform(x).unsqueeze(0)
        generated = self.model.generate(x)
        promt = open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")
        return promt

    def encode(self, x):
        emb = self.st_model.encode(x)
        return emb

    def forward(self, x):
        text = self.extract(x)
        emb = self.encode(text)
        return emb
