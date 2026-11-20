import os
import tempfile
import unicodedata
from pathlib import Path

import torch
import torch.nn.functional as F

from config import GALLERY_PATH, WEIGHT
from src.extract_keypoints import extract_keypoints
from src.utils import load_trained_vsl_model, make_features


def add_sign(video, label):
    label = unicodedata.normalize("NFC", str(label).strip())
    if not label:
        raise ValueError("Tên ký hiệu không được để trống.")
    if len(label) > 80:
        raise ValueError("Tên ký hiệu không được dài quá 80 ký tự.")
    if any(unicodedata.category(char) == "Cc" for char in label):
        raise ValueError("Tên ký hiệu không được chứa ký tự điều khiển.")

    model, device, _ = load_trained_vsl_model(WEIGHT)
    sequence = extract_keypoints(video).reshape(60, 67 * 2)
    features = make_features(
        torch.from_numpy(sequence).float().unsqueeze(0)
    ).to(device)
    with torch.no_grad():
        new_embedding = F.normalize(model(features), p=2, dim=1).cpu()

    gallery_path = Path(GALLERY_PATH)
    gallery_path.parent.mkdir(parents=True, exist_ok=True)
    if gallery_path.exists():
        gallery = torch.load(gallery_path, map_location="cpu")
        if not isinstance(gallery, dict):
            raise ValueError("Gallery hiện tại không đúng định dạng.")
        vocab = gallery.get("vocab")
        embeddings = gallery.get("gallery_emb")
        if isinstance(vocab, str):
            vocab = [vocab]
        if not isinstance(vocab, list) or not all(
            isinstance(existing_label, str) for existing_label in vocab
        ):
            raise ValueError("Danh sách nhãn trong gallery không hợp lệ.")
        if not isinstance(embeddings, torch.Tensor):
            raise ValueError("Embedding trong gallery không hợp lệ.")
        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)
        if (
            embeddings.ndim != 2
            or embeddings.shape[0] != len(vocab)
            or embeddings.shape[1] != new_embedding.shape[1]
        ):
            raise ValueError("Kích thước gallery không khớp với model.")
        vocab.append(label)
        embeddings = torch.cat(
            (embeddings, new_embedding.to(dtype=embeddings.dtype)), dim=0
        )
    else:
        vocab = [label]
        embeddings = new_embedding

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=gallery_path.parent,
            prefix=gallery_path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        torch.save(
            {"vocab": vocab, "gallery_emb": embeddings},
            temporary_path,
        )
        os.replace(temporary_path, gallery_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return label
