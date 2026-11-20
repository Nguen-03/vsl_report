from pathlib import Path
WEIGHT = Path(r'E:\Report\DL\NgonNguKyHieu\finetune\best_vsl_model.pth')
PRETRAIN_PATH = Path('E:\Report\DL\weight\last.pth')
DATASET = Path('E:\Report\DL\dataset')
LABELS = Path('E:\Report\DL\dataset\label.csv')
LABEL_MAP = Path('E:\Report\DL\dataset\label_map.json')
KEYPOINTS_PROCESSED = Path('E:\Report\DL\dataset\keypoints_processed')
SAMPLE_LABEL = Path('E:\Report\DL\dataset\sample_label.csv')
SUPPORT_SET = DATASET / 'support_set'
KEYPOINTS_RAW = Path('E:\Report\DL\dataset\keypoints')
VI_KEYPOINTS = Path(r'E:\Report\DL\dataset\vi_keypoints')
NGUYEN_KEYPOINTS = Path(r'E:\Report\DL\dataset\nguyen_keypoints')
GALLERY_PATH = Path(r'E:\Report\DL\NgonNguKyHieu\finetune\vsl_gallery.pt')
FRAMES, JOINTS, DIMS = 60, 67, 2
COORDINATE_DIM = JOINTS * DIMS
FACE_CONNECTION = [
    (0,1),(1,2), (2,3), (3,7), (0,4), (4,5), (5,6), (6,8), (9,10)
]

ARMS_CONNECTION = [
    (11,13), (13,15), (15,17), (15,19), (15,21),
    (12,14), (14,16), (16, 18), (16,20), (16, 22),
]
BODY_CONNECTION = [
    (11, 23),
    (12, 24),
    (23, 24)
]
HAND_CONNECTIONS = [
    # Cổ tay nối với gốc của 5 ngón
    (0, 1), (0, 5), (0, 17),
    (5,9), (9,13), (13,17),
    # Ngón cái (Thumb)
    (1, 2), (2, 3), (3, 4),
    # Ngón trỏ (Index Finger)
    (5, 6), (6, 7), (7, 8),
    # Ngón giữa (Middle Finger)
    (9, 10), (10, 11), (11, 12),
    # Ngón áp út (Ring Finger)
    (13, 14), (14, 15), (15, 16),
    # Ngón út (Pinky)
    (17, 18), (18, 19), (19, 20)
]

left, right = 25, 46
HAND_1 = [(u + left, v + left) for (u, v) in HAND_CONNECTIONS]
HAND_2 = [(u + right, v + right) for (u, v) in HAND_CONNECTIONS]
FACE_JOINTS = sorted({joint for edge in FACE_CONNECTION for joint in edge})
BODY_JOINTS = sorted({joint for edge in BODY_CONNECTION for joint in edge})
ARMS_JOINTS = sorted({joint for edge in ARMS_CONNECTION for joint in edge})
HAND_1_JOINTS = sorted({joint for edge in HAND_1 for joint in edge})
HAND_2_JOINTS = sorted({joint for edge in HAND_2 for joint in edge})
SELECTED_JOINTS = sorted(set(FACE_JOINTS + BODY_JOINTS + ARMS_JOINTS + HAND_1_JOINTS + HAND_2_JOINTS))
JOINT_TO_COMPACT = {joint: index for index, joint in enumerate(SELECTED_JOINTS)}
LEFT_HAND_INDICES = [JOINT_TO_COMPACT[joint] for joint in HAND_1_JOINTS]
RIGHT_HAND_INDICES = [JOINT_TO_COMPACT[joint] for joint in HAND_2_JOINTS]
JOINTS = len(SELECTED_JOINTS)
HAND_FEATURE_START = min(LEFT_HAND_INDICES + RIGHT_HAND_INDICES) * DIMS
LOCAL_HAND_DIM = (len(LEFT_HAND_INDICES) + len(RIGHT_HAND_INDICES)) * DIMS
STREAM_DIM = COORDINATE_DIM + LOCAL_HAND_DIM  # Chiều của mỗi luồng (Position hoặc Velocity)
if __name__ == '__main__':
    pass