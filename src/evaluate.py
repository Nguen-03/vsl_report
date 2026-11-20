import concurrent.futures
import re
import unicodedata as ud
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from model.embedder import VSL_Embedder
from model.ArcFace import ArcMarginProduct
from model.siamese import SiameseNetwork
from utils import *
from config import *

@torch.no_grad()
def evaluate_post_train(weight_path="/kaggle/working/best_vsl_model.pth", n_episodes=2000, k_query=5, seed=42):
    model, device, check_point = load_trained_vsl_model(weight_path)

    # 2. Nạp dữ liệu & trích xuất embedding 1 lần
    sup_x, sup_y, val_x, val_y, vocab = load_fewshot_data()

    def encode(x, batch_size=256):
        feats = make_features(x)
        outs = [
            model(feats[i : i + batch_size].to(device)).float()
            for i in range(0, len(feats), batch_size)
        ]
        return F.normalize(torch.cat(outs, dim=0), p=2, dim=1)

    gallery_emb = encode(sup_x)  # (N_classes, 512)
    query_emb = encode(val_x)    # (N_queries, 512)
    sup_y = sup_y.to(device)
    val_y = val_y.to(device)

    # Ma trận tương đồng toàn cục (N_queries, N_classes)
    sim_matrix = query_emb @ gallery_emb.T

    # Gom chỉ số query theo từng lớp hợp lệ
    valid_mask = torch.isin(sup_y, val_y)
    valid_gal_idx = torch.where(valid_mask)[0].cpu().numpy()
    valid_cls_ids = sup_y[valid_mask].cpu().numpy()
    queries_by_cls = {int(c): torch.where(val_y == int(c))[0] for c in valid_cls_ids}

    print(f"=== KẾT QUẢ ĐÁNH GIÁ SAU TRAIN ({Path(weight_path).name} | epoch {check_point.get('epoch')}) ===")
    print(f"Tổng số lớp test chưa từng thấy: {len(valid_cls_ids)} | Tổng mẫu query: {len(val_y)}\n")

    # 3. Đánh giá N-way 1-shot (5-way, 10-way, 20-way)
    for n_way in [5, 10, 20]:
        if len(valid_gal_idx) < n_way:
            continue
        rng = np.random.default_rng(seed)
        ep_accs = []

        for _ in range(n_episodes):
            chosen_pos = rng.choice(len(valid_gal_idx), size=n_way, replace=False)
            chosen_gal = torch.as_tensor(valid_gal_idx[chosen_pos], device=device)
            chosen_cls = valid_cls_ids[chosen_pos]

            q_indices, targets = [], []
            for local_label, cls_id in enumerate(chosen_cls):
                pool = queries_by_cls[int(cls_id)]
                take = min(len(pool), k_query)
                perm = rng.choice(len(pool), size=take, replace=False)
                q_sel = pool[torch.as_tensor(perm, device=device)]
                q_indices.append(q_sel)
                targets.append(torch.full((take,), local_label, dtype=torch.long, device=device))

            q_indices = torch.cat(q_indices)
            targets = torch.cat(targets)

            ep_sims = sim_matrix[q_indices][:, chosen_gal]
            acc = (ep_sims.argmax(dim=1) == targets).float().mean().item()
            ep_accs.append(acc * 100.0)

        mean_acc = float(np.mean(ep_accs))
        ci95 = 1.96 * float(np.std(ep_accs, ddof=1)) / np.sqrt(n_episodes)
        print(f"  • {n_way:2d}-way 1-shot ({n_episodes} episodes): {mean_acc:.2f}% ± {ci95:.2f}%")

    # 4. Đánh giá Full-way 1-shot (toàn bộ 199 lớp cùng lúc)
    topk_idx = sim_matrix.topk(k=5, dim=1).indices
    preds = sup_y[topk_idx]
    full_top1 = (preds[:, 0] == val_y).float().mean().item() * 100.0
    full_top5 = (preds == val_y[:, None]).any(dim=1).float().mean().item() * 100.0
    print(f"  • Full {len(sup_y)}-way 1-shot:          Top-1 = {full_top1:.2f}% | Top-5 = {full_top5:.2f}%")

# def official_test():
    
if __name__ == '__main__':
    evaluate_post_train(WEIGHT)
    data = torch.load(r"E:\Report\DL\NgonNguKyHieu\finetune\vsl_gallery.pt", map_location="cpu")
    stats = compute_embedding_dispersion(data["gallery_emb"])
    print(stats)
    # researcher 
    # p7skn!dkah