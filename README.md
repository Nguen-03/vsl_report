# Dấu Tay

Giao diện nhận diện ngôn ngữ ký hiệu tiếng Việt từ video hoặc camera.

## Chạy giao diện

1. Giải nén toàn bộ ZIP. Cần Python 3 và trình duyệt Chrome hoặc Edge.
2. Trên Windows, mở `Chay_Giao_Dien.bat`. Hoặc mở terminal trong thư mục giải nén và chạy `python server.py` (`python3 server.py` trên macOS/Linux).
3. Mở http://127.0.0.1:8000 nếu trình duyệt chưa tự mở. Giữ cửa sổ Python trong lúc sử dụng. Nếu cổng 8000 đang bận, chạy `python server.py --port 8001`.
4. Chọn **Tải video**, chọn tệp rồi bấm **Bắt đầu nhận diện**. Với **Camera**, bấm **Bật camera**, cấp quyền, bấm **Bắt đầu** để ghi và **Kết thúc** để gửi video nhận diện.

Chọn **Thêm ký hiệu** để nhập tên, bật camera, rồi bấm **Bắt đầu/Kết thúc** để lưu một mẫu vào gallery. Chức năng này cần model và gallery hợp lệ; mẫu mới được thêm vào, không thay thế các mẫu hiện có.

Khi chạy thử, kết quả “Xin chào” được đánh dấu **Minh họa**. Lịch sử lưu tối đa 50 kết quả gần nhất trong trình duyệt. Lịch sử lưu nhãn, thời điểm và tên nguồn; video được máy chủ giữ tạm trong lúc xử lý.

## Nối model đã huấn luyện

1. Sao chép `model_adapter.example.py` thành `model_adapter.py` cạnh `server.py`.
2. Đặt checkpoint, mã suy luận và bảng ánh xạ lớp của dự án vào thư mục phù hợp. Cài các thư viện mà mô hình cần trong cùng môi trường Python dùng để mở server.
3. Nạp model một lần và gọi hàm suy luận từ `predict(video_path, *, source)`. Trả về `{"label": "Xin chào", "confidence": 0.92}`. Nếu model chỉ có nhãn thì trả `confidence: None`. Giữ cách lấy frame, trích xuất MediaPipe, chuẩn hóa, số bước thời gian và thứ tự lớp như khi huấn luyện.
4. Khởi động lại server. Khi adapter được nạp, góc trên sẽ hiện **Đã kết nối model**. Thử một video đã biết nhãn để kiểm tra đầy đủ đường suy luận.

Camera gửi WebM hoặc MP4 tùy trình duyệt. Hàm suy luận cần đọc định dạng nhận được hoặc chuyển mã trước bước lấy frame.

Máy chủ cung cấp `GET /api/health`, `POST /api/predict?source=video&name=clip.mp4` và `POST /api/add-sign?label=...`. Các API POST nhận byte video trực tiếp. API nhận diện trả `label`, `confidence`, `demo`; API thêm ký hiệu trả nhãn đã lưu. Máy chủ mặc định chỉ lắng nghe tại localhost.
