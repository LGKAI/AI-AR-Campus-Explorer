# AI AR Campus Explorer

**Đồ án môn học:** Tư duy tính toán cho AI (Computational Thinking for AI)  
**Trường Đại học:** Trường Đại học Khoa học Tự nhiên - ĐHQG TP.HCM (HCMUS)  

🎥 **Video Demo Dự Án:** [Xem trên YouTube](https://youtu.be/bCx2UIy9N5Y?si=EkbJ5mAB5oKfTRJj)

---

## Giới thiệu tổng quan

**AI AR Campus Explorer** là một nền tảng bản đồ học thuật thông minh và trợ lý số toàn diện dành cho sinh viên trường Đại học Khoa học Tự nhiên (HCMUS). 

Hệ thống kết hợp nhiều công nghệ AI tiên tiến nhằm mang đến trải nghiệm định hướng hiện đại nhất:
- **Đồ thị tri thức & Mạng nơ-ron đồ thị (GNN - Graph Neural Network)** kết hợp thuật toán tối ưu để tìm đường đi thông minh trong khuôn viên trường.
- **Thực tế tăng cường (AR Camera WebXR Navigation)** với HUD định vị la bàn số (Compass) và GPS thời gian thực dẫn đường trực tiếp trên không gian thực tế qua camera điện thoại.
- **Trợ lý ảo học vụ STELLAR-RAG (Retrieval-Augmented Generation)** sử dụng mô hình ngôn ngữ lớn LLM (Ollama Qwen2.5) hỗ trợ giải đáp quy chế và chỉ đường bằng giọng nói (Web Speech API).
- **Thị giác máy tính (Computer Vision - YOLOv8)** nhận diện khuôn mặt và đăng nhập sinh viên bằng FaceID nhanh chóng, bảo mật.
- **Hệ tư vấn AI & Khám phá địa điểm (Tiktok Swipe Explore)** gợi ý các toà nhà, tiện ích phù hợp theo sở thích, lịch sử di chuyển kèm giao diện thẻ vuốt tương tác cao.
- **Giao diện Glassmorphism & Responsive Toàn Diện**: Thiết kế sang trọng, tương thích hoàn hảo mọi thiết bị di động (hỗ trợ tai thỏ / Dynamic Island), menu 3 gạch linh hoạt cho smartphone.

---

## Công nghệ & Kỹ thuật áp dụng

### 1. Trí tuệ Nhân tạo (AI & Machine Learning)
- **Mô hình Ngôn ngữ Lớn (LLMs):** Tích hợp Ollama (Qwen2.5) cho hệ thống **RAG Chatbot** truy xuất thông tin quy chế đào tạo, sổ tay sinh viên HCMUS.
- **Thị giác Máy tính (Computer Vision):** Mô hình **YOLOv8** phát hiện và nhận diện khuôn mặt sinh viên qua camera thời gian thực để đăng nhập FaceID.
- **Mạng nơ-ron Đồ thị & Tìm đường (GNN & Pathfinding):** Biểu diễn khuôn viên thành không gian đồ thị nút-cạnh, huấn luyện mô hình PyTorch GNN kết hợp thuật toán A* để tối ưu hoá lộ trình theo điều kiện thực tế (thời tiết nắng/mưa, có mái che, hỗ trợ xe lăn).
- **Hệ thống Gợi ý (Recommendation Engine):** Dự đoán địa điểm tiếp theo sinh viên quan tâm dựa trên lịch sử di chuyển và khung giờ.

### 2. Backend & Máy chủ API
- **FastAPI (Python):** Framework RESTful API hiệu năng cao, bất đồng bộ (Asynchronous).
- **WebSockets:** Kênh truyền tải dữ liệu trạng thái thời gian thực giữa thiết bị và server.
- **Bảo mật:** Xác thực người dùng qua JWT (JSON Web Token), mã hóa mật khẩu và tích hợp Firebase Database.

### 3. Frontend & Trải nghiệm Người dùng
- **HTML5, Vanilla CSS & TailwindCSS:** Thiết kế giao diện hiện đại theo phong cách Glassmorphism và Cyber-glow.
- **WebXR API & Leaflet.js:** Bản đồ 2D tương tác kết hợp AR Camera 3D Overlay.
- **Web Speech API:** Nhận diện giọng nói (Voice-to-Text) để trò chuyện trực tiếp với Chatbot.
- **Hỗ trợ Safe Area:** Sử dụng `env(safe-area-inset-top)` và `env(safe-area-inset-bottom)` bảo đảm hiển thị tràn viền hoàn mỹ trên iOS & Android.

### 4. Triển khai & Hạ tầng (Deployment)
- **Ngrok Tunnel:** Mở đường hầm an toàn đưa Backend Local ra Internet với cơ chế Bypass Header tự động (`ngrok-skip-browser-warning`).
- **Netlify:** Đám mây lưu trữ và phân phối ứng dụng Frontend toàn cầu.
- **Cơ chế Tự động Nhận diện Môi trường (Smart Environment Detection):** Tự chuyển đổi kết nối giữa `localhost:8000` (khi chạy nội bộ) và domain Ngrok (khi chạy qua Netlify trên điện thoại).

---

## Cấu trúc thư mục dự án

```text
source/
├── backend/
│   ├── ai/
│   │   ├── information_chatbot/       # Module RAG Chatbot (Hỏi đáp quy chế HCMUS)
│   │   ├── local_map/                 # Module Bản đồ 2D & API AR Camera Navigation
│   │   │   ├── data/                  # Toạ độ các toà nhà & đồ thị nút di chuyển
│   │   │   └── router.py              # FastAPI Router cung cấp dữ liệu bản đồ
│   │   ├── recommendation_system/     # Module GNN & Thuật toán tìm đường thông minh
│   │   │   ├── engine/                # Mô hình GNN PyTorch (.pth) & logic định tuyến
│   │   │   └── recommender.py         # Hệ thống gợi ý địa điểm cá nhân hoá
│   │   └── face_guard/                # Module YOLOv8 Face Recognition (FaceID)
│   ├── core/
│   │   ├── config.py                  # Cấu hình hệ thống & Firebase
│   │   └── security.py                # Xử lý JWT Token & mã hoá mật khẩu
│   ├── models/
│   │   └── schemas.py                 # Khai báo Schema Pydantic cho dữ liệu Request/Response
│   ├── routers/
│   │   ├── users.py                   # API Đăng ký / Đăng nhập / FaceID
│   │   ├── locations.py               # API Danh sách toà nhà, phòng học
│   │   ├── history.py                 # API Lịch sử di chuyển của sinh viên
│   │   ├── recommendation.py          # API Định tuyến GNN & Gợi ý điểm đến
│   │   ├── chat.py                    # API RAG Chatbot
│   │   └── ws.py                      # API WebSocket kết nối thời gian thực
│   ├── main.py                        # Điểm khởi chạy chính của FastAPI Backend
│   ├── yolov8n.pt                     # Trọng số mô hình YOLOv8
│   ├── .env                           # Biến môi trường
│   └── requirements.txt               # Danh sách thư viện Python
├── frontend/
│   ├── img/                           # Logo, hình ảnh tư liệu toà nhà HCMUS
│   ├── local_map/                     # Script AR WebXR (ar-integration.js) & Leaflet
│   ├── index.html                     # Giao diện chính của ứng dụng
│   ├── style.css                      # Tuỳ chỉnh CSS, hiệu ứng ánh sáng, Safe Area
│   └── script.js                      # Logic điều khiển, API kết nối, WebSocket, Responsive Nav
├── ngrok.exe                          # Tiện ích Ngrok Tunnel
└── README.md                          # Tài liệu hướng dẫn dự án
```

---

## Hướng dẫn cài đặt và chạy ứng dụng

Ứng dụng áp dụng kiến trúc: **Backend chạy trên máy tính** (để khai thác sức mạnh AI, mô hình PyTorch, YOLOv8 và Ollama) và **Frontend chạy trên máy tính hoặc đám mây Netlify** (để truy cập trên điện thoại mọi lúc mọi nơi).

### Bước 1: Khởi động Backend (Máy chủ AI)
*Yêu cầu môi trường:* Python 3.10 trở lên.

1. Mở Terminal và điều hướng vào thư mục `backend`:
   ```bash
   cd backend
   ```
2. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```
3. Đảm bảo phần mềm **Ollama** đang chạy trên máy tính (nếu sử dụng tính năng Chatbot RAG).
4. Khởi chạy máy chủ FastAPI:
   ```bash
   uvicorn main:app --reload
   ```
   *Máy chủ sẽ chạy tại địa chỉ `http://127.0.0.1:8000`.*

---

### Bước 2: Kích hoạt đường hầm Ngrok (Để mở kết nối cho điện thoại)
1. Mở một cửa sổ Terminal thứ hai (tại thư mục `source`):
   ```powershell
   .\ngrok http 8000
   ```
   *(Nếu dùng macOS/Linux: `ngrok http 8000`).*
2. Ngrok sẽ kích hoạt đường hầm Public HTTPS (ví dụ: `https://essential-snipping-parcel.ngrok-free.dev`).
3. **Giữ nguyên cả 2 cửa sổ Terminal này** trong suốt quá trình sử dụng.

---

### Bước 3: Trải nghiệm ứng dụng

#### Cách 1: Chạy trực tiếp trên máy tính (Local)
- Mở thư mục `frontend` bằng VS Code và chọn **Open with Live Server** (hoặc mở file `index.html` trong trình duyệt).
- Hệ thống sẽ tự động phát hiện `localhost` và kết nối trực tiếp đến `http://127.0.0.1:8000`.

#### Cách 2: Chạy trên điện thoại di động qua Netlify
1. Mã nguồn đã được cấu hình sẵn domain Ngrok cố định trong `frontend/script.js`.
2. Kéo thả thư mục `frontend` lên **[Netlify Drop](https://app.netlify.com/drop)** (hoặc liên kết repository GitHub để Netlify tự động deploy).
3. Mở đường link Netlify trên trình duyệt điện thoại (Safari / Chrome).
4. Trải nghiệm đầy đủ các tính năng:
   - Đăng nhập FaceID qua Camera điện thoại.
   - Trải nghiệm chỉ đường thực tế ảo qua AR Camera WebXR.
   - Hỏi đáp bằng giọng nói với Trợ lý ảo RAG Chatbot.
   - Thử nghiệm tìm đường GNN và chạy mô phỏng di chuyển GPS.