
import concurrent.futures
import re
import unicodedata as ud
from pathlib import Path
import numpy as np
import torch
import math
import torch.nn.functional as F
from model.embedder import VSL_Embedder
from model.ArcFace import ArcMarginProduct
from model.siamese import SiameseNetwork
from config import *
def _normalize_hand_wrist(hand_joints):
    """Chuẩn hóa cục bộ 21 khớp bàn tay theo cổ tay (0) và chiều dài lòng bàn tay (0 -> 9)."""
    present = hand_joints.ne(0).any(dim=-1, keepdim=True)
    wrist = hand_joints[..., 0:1, :]
    middle_mcp = hand_joints[..., 9:10, :]

    palm_size = torch.linalg.norm(middle_mcp - wrist, dim=-1, keepdim=True).clamp(min=1e-4)
    return ((hand_joints - wrist) / palm_size) * present


def _compute_velocity(norm_joints):
    """Tính vận tốc giữa 2 frame liên tiếp chỉ khi cả 2 frame đều có dữ liệu khớp."""
    present = norm_joints.ne(0).any(dim=-1, keepdim=True)
    vel = torch.zeros_like(norm_joints)
    valid_transition = present[..., 1:, :, :] & present[..., :-1, :, :]
    vel[..., 1:, :, :] = (norm_joints[..., 1:, :, :] - norm_joints[..., :-1, :, :]) * valid_transition
    return vel

def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        for k in ("sequence", "keypoints", "data", *z.files):
            if k in z:
                a = np.asarray(z[k], dtype=np.float32)
                if a.ndim >= 2:
                    if a.shape == (FRAMES, JOINTS, 3):
                        a = a[:, SELECTED_JOINTS, :DIMS]
                    elif a.shape == (FRAMES, JOINTS * 3):
                        a = a.reshape(FRAMES, JOINTS, 3)[:, SELECTED_JOINTS, :DIMS]
                    elif a.shape == (FRAMES, JOINTS, 3):
                        a = a[:, :, :DIMS]
                    elif a.shape == (FRAMES, JOINTS * 3):
                        a = a.reshape(FRAMES, JOINTS, 3)[:, :, :DIMS]
                    elif a.shape == (FRAMES, JOINTS * DIMS):
                        a = a.reshape(FRAMES, JOINTS, DIMS)
                    elif a.shape != (FRAMES, JOINTS, DIMS):
                        raise ValueError(f"{path} has invalid keypoint shape {a.shape}")
                    return a.reshape(FRAMES, JOINTS * DIMS)
    raise ValueError(f"{path}: không có mảng số hợp lệ, các key = {z.files}")

def make_features(seq):
    if seq.ndim not in (2, 3):
        raise ValueError(f"Expected input with selected-joint coordinates, got {tuple(seq.shape)}")
    if seq.shape[-1] == STREAM_DIM * 2:
        return seq
    # Tự động cắt lấy 134 tọa độ gốc nếu MyDataset trả về tensor 268 chiều cũ
    if seq.shape[-1] == COORDINATE_DIM * 2:
        seq = seq[..., :COORDINATE_DIM]
    elif seq.shape[-1] != COORDINATE_DIM:
        raise ValueError(
            f"Expected {COORDINATE_DIM} coordinate features or "
            f"{STREAM_DIM * 2} normalized position-and-velocity features, got {seq.shape[-1]}"
        )

    joints = seq.reshape(*seq.shape[:-1], JOINTS, DIMS)

    # 1. Vận tốc toàn cục trên hệ quy chiếu vai có sẵn
    global_vel = _compute_velocity(joints)

    # 2. Chuẩn hóa cục bộ 2 bàn tay theo cổ tay & kích thước lòng bàn tay
    lhand = joints[..., LEFT_HAND_INDICES, :]
    rhand = joints[..., RIGHT_HAND_INDICES, :]

    lhand_norm = _normalize_hand_wrist(lhand)
    rhand_norm = _normalize_hand_wrist(rhand)
    lhand_vel = _compute_velocity(lhand_norm)
    rhand_vel = _compute_velocity(rhand_norm)

    # 3. Ghép luồng Position (218) và Velocity (218) -> 436 chiều
    pos_stream = torch.cat([
        seq,
        lhand_norm.reshape(*seq.shape[:-1], -1),
        rhand_norm.reshape(*seq.shape[:-1], -1),
    ], dim=-1)

    vel_stream = torch.cat([
        global_vel.reshape_as(seq),
        lhand_vel.reshape(*seq.shape[:-1], -1),
        rhand_vel.reshape(*seq.shape[:-1], -1),
    ], dim=-1)

    return torch.cat((pos_stream, vel_stream), dim=-1)


def _norm_name(s):
    return re.sub(r"\s+", " ", ud.normalize("NFC", str(s).strip())).lower()

def load_fewshot_data(support_dir=SUPPORT_SET, query_dir=NGUYEN_KEYPOINTS):
    sup_dirs = [d for d in sorted(Path(support_dir).iterdir()) if d.is_dir()]
    vocab = sorted({_norm_name(d.name) for d in sup_dirs})
    word2id = {w: i for i, w in enumerate(vocab)}

    def read_folder(root, per_class=None):
        tasks = []
        for d in sorted(Path(root).iterdir()):
            if not d.is_dir():
                continue
            cls_id = word2id.get(_norm_name(d.name))
            if cls_id is None:
                continue
            files = sorted(d.glob("*.npz"))
            if per_class is not None:
                files = files[:per_class]
            for f in files:
                tasks.append((f, cls_id))

        with concurrent.futures.ThreadPoolExecutor(max_workers=24) as ex:
            seqs = list(ex.map(load_npz, [t[0] for t in tasks]))
        labels = [t[1] for t in tasks]
        return torch.from_numpy(np.stack(seqs)).float(), torch.tensor(labels, dtype=torch.long)

    sup_x, sup_y = read_folder(support_dir, per_class=1)
    val_x, val_y = read_folder(query_dir, per_class=None)

    keep = torch.isin(val_y, sup_y)
    return sup_x, sup_y, val_x[keep], val_y[keep], vocab


def load_trained_vsl_model(weight_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Bước 1: Đọc file checkpoint lên CPU trước
    checkpoint = torch.load(weight_path, map_location="cpu")

    # Lấy đúng dictionary chứa trọng số (hỗ trợ cả file lưu dict hoặc lưu trực tiếp state_dict)
    raw_state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

    # Bước 2: Gỡ bỏ tiền tố "module." nếu file được train bằng nn.DataParallel
    clean_state_dict = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in raw_state_dict.items()
    }

    # Bước 3: Khởi tạo vỏ mạng và nạp trọng số với strict=True
    embedder = VSL_Embedder(stream_dim=STREAM_DIM, hidden_dim=256, embed_dim=512)
    model = SiameseNetwork(encoder=embedder).to(device)
    model.load_state_dict(clean_state_dict, strict=True)

    # Bước 4: Chuyển sang chế độ đánh giá (khóa Dropout)
    model.eval()
    return model, device, clean_state_dict

@torch.no_grad()
def compute_embedding_dispersion(embeddings: torch.Tensor) -> dict:
    """
    Tính độ phân tán (Vector Std Mean) của tập vector đặc trưng trên mặt cầu đơn vị.
    
    Args:
        embeddings: Tensor kích thước (N, D), ví dụ (N, 512)
    """
    # 1. Đảm bảo tất cả vector đã chuẩn hóa L2 nằm trên siêu cầu đơn vị (||z||_2 = 1)
    z = F.normalize(embeddings.float(), p=2, dim=1)
    num_samples, dim = z.shape

    # 2. Tính độ lệch chuẩn (Std) dọc theo batch cho từng chiều trong 512 chiều -> (512,)
    per_dim_std = z.std(dim=0, unbiased=False)

    # 3. Lấy trung bình trên toàn bộ 512 chiều -> Vector Std Mean
    vector_std_mean = float(per_dim_std.mean().item())

    # 4. So sánh với ngưỡng lý tưởng của phân bố đẳng hướng trên mặt cầu D chiều: 1 / sqrt(D)
    ideal_std = 1.0 / math.sqrt(dim)
    isotropy_ratio = vector_std_mean / ideal_std

    # 5. (Tùy chọn) Tính thêm độ tương đồng Cosine trung bình giữa các cặp từ khác nhau
    # Giá trị càng gần 0 chứng tỏ các từ đẩy nhau ra càng đều
    sim_matrix = z @ z.T
    mask = ~torch.eye(num_samples, dtype=torch.bool, device=z.device)
    mean_pairwise_cos = float(sim_matrix[mask].mean().item()) if num_samples > 1 else 0.0

    return {
        "vector_std_mean": round(vector_std_mean, 4),
        "ideal_std": round(ideal_std, 4),
        "isotropy_ratio": round(isotropy_ratio, 4),
        "mean_pairwise_cosine": round(mean_pairwise_cos, 4),
    }
if __name__ == '__main__':
    pass