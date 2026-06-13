# AI AR Campus Explorer

Đồ án Tư duy tính toán cho AI

Video Demo: https://youtu.be/bCx2UIy9N5Y?si=EkbJ5mAB5oKfTRJj

Interactive Prototype: https://lgkai.github.io/AACE-Interactive-Prototype/

AI AR Campus Explorer là một hệ thống bản đồ học thuật tiên tiến dành cho sinh viên trường Đại học Khoa học Tự nhiên (HCMUS). 
Hệ thống kết hợp nhiều công nghệ AI hiện đại như **Knowledge Graph (GNN)** để tìm đường, **RAG Chatbot** để hỏi đáp thông tin quy chế, **WebXR** để trải nghiệm AR 3D Hologram, và **Face Recognition** để quản lý tài khoản.

## Cấu trúc thư mục

```text
source/
├── backend/
│   ├── ai/
│   │   ├── information_chatbot/   # Module STELLAR-RAG (Hỏi đáp quy chế) do Kin phụ trách
│   │   ├── local_map/             # Module Bản đồ 2D (Leaflet) & API AR Camera Navigation do Hiếu phụ trách
│   │   │   ├── data/              # Chứa file toạ độ toà nhà (buildings.json) & điểm gốc (nav_graph.json)
│   │   │   └── router.py          # FastAPI Router cấp dữ liệu bản đồ cục bộ
│   │   ├── recommendation_system/ # Module GNN & Pathfinding (Tìm đường) do Khoa phụ trách
│   │   └── face_guard/            # Module Face Recognition (Xác thực khuôn mặt) do Khanh phụ trách
│   ├── core/
│   │   ├── config.py              # Cấu hình kết nối Firebase
│   │   ├── security.py            # Xử lý JWT Token và mã hóa mật khẩu
│   │   └── serviceAccountKey.json # Chứa khoá bí mật giao tiếp với Firebase DB
│   ├── models/
│   │   └── schemas.py             # Khai báo cấu trúc dữ liệu đầu vào/ra (Pydantic)
│   ├── routers/
│   │   ├── users.py               # API Quản lý tài khoản và xác thực (Login/Register/FaceID)
│   │   ├── locations.py           # API Truy xuất dữ liệu tòa nhà, phòng học
│   │   ├── history.py             # API Truy xuất dữ liệu lịch sử di chuyển của User
│   │   ├── recommendation.py      # API cho hệ thống bản đồ (GNN, Tìm đường, Đề xuất AI)
│   │   ├── chat.py                # API cho hệ thống RAG Chatbot
│   │   └── ws.py                  # API WebSocket truyền dữ liệu thời gian thực
│   ├── main.py                    # File cấu hình gốc của ứng dụng FastAPI
│   ├── yolov8n.pt                 # Mô hình nhận diện YOLOv8
│   ├── .env                       # File cấu hình hệ thống cho các thư viện
│   └── requirements.txt           # Danh sách các thư viện cần thiết
└── frontend/
    ├── img/                       # Thư mục chứa hình ảnh tài nguyên
    ├── local_map/                 # Thư mục chứa CSS và script AR Camera (ar-integration.js, leaflet)
    ├── index.html                 # Giao diện của ứng dụng
    ├── style.css                  # Thiết kế giao diện
    └── script.js                  # Logic xử lý giao diện, bản đồ, gọi API, Web Speech API
```

## Các tính năng nổi bật
1. **Tìm đường thông minh (GNN & A*):** Tìm kiếm lộ trình nhanh nhất, lộ trình có mái che, phù hợp thời tiết.
2. **Trợ lý ảo AI RAG Chatbot:** Hỗ trợ sinh viên tra cứu quy chế, tìm đường bằng giọng nói (Voice-to-Text).
3. **Thực tế ảo tăng cường (WebXR AR):** Nhấn vào đỉnh đồ thị trên bản đồ để xem mô hình 3D Hologram.
4. **Xác thực khuôn mặt:** Đăng nhập an toàn và nhanh chóng bằng FaceID.
5. **Đề xuất cá nhân hoá:** Đề xuất điểm đến dựa trên lịch sử ghé thăm và khung giờ.

## Hướng dẫn cài đặt và chạy hệ thống

### 1. Khởi động Backend (Máy chủ API)
Đảm bảo bạn đã cài đặt Python 3.10+ trên máy tính.

Mở terminal và trỏ vào thư mục `backend`:
```bash
cd backend
```

Cài đặt tất cả các thư viện cần thiết:
```bash
pip install -r requirements.txt
```
*(Lưu ý: Do tích hợp nhiều module AI như PyTorch, Ultralytics, Ollama, việc cài đặt có thể mất vài phút tuỳ vào tốc độ mạng)*

Chạy máy chủ FastAPI:
```bash
uvicorn main:app --reload
```
Máy chủ sẽ chạy tại địa chỉ: `http://127.0.0.1:8000`

### 2. Khởi động Frontend (Giao diện người dùng)
Frontend được code thuần bằng HTML, JS và nạp TailwindCSS qua CDN để đảm bảo tính gọn nhẹ.

Cách chạy nhanh nhất:
1. Mở thư mục `frontend`.
2. Dùng extension **Live Server** trên VSCode.
3. Hoặc mở trực tiếp file `index.html` trên trình duyệt Chrome/Edge/Firefox.

> **Lưu ý:** Để tính năng thu âm (Web Speech API) và Webcam (FaceID) hoạt động, bạn cần chạy Frontend trên một Local server (VD: `http://127.0.0.1:5500`) thay vì mở file dạng `file:///...` do chính sách bảo mật của trình duyệt.

### 3. Hướng dẫn test trên Điện thoại (Cùng mạng Wi-Fi)
Để sử dụng đồ án trên điện thoại thật (cần thiết cho chức năng GPS và Camera AR), bạn cần làm theo các bước sau:

**Bước 1: Khởi động Backend cho phép mạng ngoài truy cập**
Thay vì chạy `127.0.0.1`, bạn hãy chạy Backend với host `0.0.0.0`:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Bước 2: Sửa IP trong Frontend**
1. Lấy địa chỉ IPv4 của Laptop (Dùng lệnh `ipconfig` trên Windows, ví dụ `192.168.1.15`).
2. Sửa toàn bộ đường dẫn gọi API trong các file JS (như `api.js`, `script.js`) từ `http://localhost:8000` thành `http://<IP_Laptop>:8000`.

**Bước 3: Vượt rào bảo mật HTTPS (Bắt buộc cho Camera & GPS)**
Trình duyệt trên điện thoại sẽ chặn Camera và GPS nếu đường dẫn truy cập là HTTP thay vì HTTPS. Bạn có 2 cách giải quyết:
- **Cách 1 (Dành cho điện thoại Android):** Mở Chrome trên Android, truy cập vào `chrome://flags/#unsafely-treat-insecure-origin-as-secure`. Điền địa chỉ IP của frontend (VD: `http://192.168.1.15:5500`) vào ô, bật **Enabled** và khởi động lại Chrome.
- **Cách 2 (Dành cho iPhone/Mọi thiết bị):** Dùng công cụ tạo hầm (tunnel) để lấy link HTTPS tạm thời. Trên laptop, chạy lệnh NodeJS:
```bash
npx localtunnel --port 5500
```
Truy cập vào đường link `https://...` vừa được cấp bằng điện thoại.
