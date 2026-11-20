"""Đổi tên thành model_adapter.py, rồi nối hàm suy luận đã huấn luyện."""
from pathlib import Path
import torch
from config import *
import unicodedata
import torch.nn.functional as F
from NgonNguKyHieu.src.evaluate import load_trained_vsl_model, make_features
from NgonNguKyHieu.src.extract_keypoints import extract_keypoints
# Nạp model một lần tại đây. Cài thư viện của model vào Python chạy server.
# Ví dụ: from inference import recognize_video

model, device, _ = load_trained_vsl_model(WEIGHT)

_gallery_data = torch.load(GALLERY_PATH, map_location=device)
VOCAB: list[str] = [unicodedata.normalize("NFC", str(w)) for w in _gallery_data["vocab"]]
GALLERY_EMB: torch.Tensor = F.normalize(_gallery_data["gallery_emb"].float().to(device), p=2, dim=1)


def reload_gallery() -> None:
    """Reload gallery vocabulary and embeddings after a new sign is added."""
    global VOCAB, GALLERY_EMB
    gallery_data = torch.load(GALLERY_PATH, map_location=device)
    VOCAB = [
        unicodedata.normalize("NFC", str(word))
        for word in gallery_data["vocab"]
    ]
    GALLERY_EMB = F.normalize(
        gallery_data["gallery_emb"].float().to(device),
        p=2,
        dim=1,
    )
def predict(video_path: Path, *, source: str) -> dict:
    """video_path là tệp tạm MP4/WebM/...; source là 'video' hoặc 'camera'."""
    # Đưa video mới vào -> rút vector query_emb (1, 512)
    sequence = extract_keypoints(video_path)
    sequence = sequence.reshape(60, 67 * 2)
    features = make_features(torch.from_numpy(sequence).float().unsqueeze(0)).to(device)
    query_emb = F.normalize(model(features), p=2, dim=1)

    # So trên không gian vector và tra ngược nhãn thật qua chỉ số index
    sims = torch.clamp((query_emb @ GALLERY_EMB.T).squeeze(0), min=-1.0, max=1.0)  # (N,)
    k = min(5, len(VOCAB))
    top_scores, top_indices = torch.topk(sims, k=k)

    top5 = [
        {
            "label": unicodedata.normalize("NFC", VOCAB[idx]),
            "cosine_distance": round(1.0 - float(score), 4),
        }
        for score, idx in zip(top_scores.tolist(), top_indices.tolist())
    ]

    best = top5[0]
    return {
        "label": best["label"],
        "cosine_distance": best["cosine_distance"],
        "top5": top5,
    }
    # raise NotImplementedError("Hãy nối hàm suy luận của model vào predict().")
