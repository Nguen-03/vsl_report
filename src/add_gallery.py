# from pathlib import Path
import torch
import numpy as np
import unicodedata
# from NgonNguKyHieu.model.embedder import VSL_Embedder
# from NgonNguKyHieu.model.siamese import SiameseNetwork
from evaluate import load_trained_vsl_model, make_features
from config import *

def add_gallery(keypoints_path, WEIGHT):
    model, device, check_point = load_trained_vsl_model(WEIGHT)
    vocab = []       # List chứa tên nhãn thật: ["An Giang", "Bắc Ninh", ...]
    embeddings = []  # List chứa tensor (1, 512) tương

    for dir in keypoints_path.iterdir():
        
        for file in list(dir.rglob('*.npz')):
            with np.load(file) as d:
                sequence = d['sequence'].reshape(60, 67, 3)
                # print(sequence.shape)
                sequence = sequence[:,:,:2]
                sequence = sequence.reshape(60, 67 * 2)
                features = make_features(torch.from_numpy(sequence).float().unsqueeze(0)).to(device)
                emb = model(features)
                label = file.parent.name
                label = unicodedata.normalize('NFC', label)
                vocab.append(label)
                embeddings.append(emb)
        
    gallery_emb = torch.cat(embeddings, dim=0)  # (N, 512)
    torch.save({"vocab": vocab, "gallery_emb": gallery_emb.cpu()}, GALLERY_PATH)
if __name__ == '__main__':
    add_gallery(KEYPOINTS_RAW, WEIGHT)