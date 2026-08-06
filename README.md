# AI AR Campus Explorer

Đồ án Tư duy tính toán cho AI

🎥 **Video Demo:** [Xem trên YouTube](https://youtu.be/bCx2UIy9N5Y?si=EkbJ5mAB5oKfTRJj)

AI AR Campus Explorer là một hệ thống bản đồ học thuật tiên tiến dành cho sinh viên trường Đại học Khoa học Tự nhiên (HCMUS). 
Hệ thống kết hợp nhiều công nghệ AI hiện đại như **Knowledge Graph (GNN)** để tìm đường, **RAG Chatbot** để hỏi đáp thông tin quy chế, **WebXR** để trải nghiệm AR 3D Hologram, và **Face Recognition** để quản lý tài khoản.

## Công nghệ và Kiến thức sử dụng

Dự án áp dụng các kiến thức chuyên sâu về Trí tuệ Nhân tạo và Phát triển Phần mềm:
- **Trí tuệ nhân tạo (AI & Machine Learning):**
  - **Mô hình ngôn ngữ lớn (LLM):** Ứng dụng Ollama (Qwen2.5) cho hệ thống RAG Chatbot hỗ trợ giải đáp quy chế học vụ.
  - **Thị giác máy tính (Computer Vision):** Ứng dụng mô hình YOLOv8 để nhận diện và trích xuất đặc trưng khuôn mặt (FaceID).
  - **Đồ thị tri thức & Thuật toán tìm kiếm (Graph & Pathfinding):** Sử dụng Graph Neural Network (GNN) và thuật toán A* để tối ưu hóa lộ trình di chuyển trong khuôn viên trường.
  - **Hệ tư vấn (Recommendation System):** Gợi ý địa điểm thông minh dựa trên lịch sử di chuyển và ngữ cảnh người dùng.
- **Backend (Máy chủ & API):**
  - Xây dựng bằng **Python** và **FastAPI** mang lại hiệu suất cao.
  - Quản lý giao tiếp thời gian thực với **WebSockets**.
  - Bảo mật xác thực qua **JWT Token**.
- **Frontend (Giao diện & Tương tác):**
  - Giao diện thuần **HTML/CSS/JS** tích hợp **TailwindCSS**.
  - **WebXR API:** Hỗ trợ hiển thị mô hình thực tế ảo tăng cường (AR 3D Hologram) ngay trên trình duyệt web.
  - **Leaflet.js:** Tích hợp bản đồ 2D tương tác.
  - **Web Speech API:** Tích hợp nhận diện giọng nói (Voice-to-Text).
- **Triển khai (Deployment):**
  - **Ngrok:** Thiết lập đường hầm (tunnel) an toàn đưa localhost API Server ra Internet.
  - **Netlify Drop:** Hosting tĩnh cho ứng dụng Frontend.

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

## Hướng dẫn triển khai và chạy hệ thống (Deployment)

Dự án này sử dụng mô hình Backend chạy cục bộ (để tận dụng sức mạnh xử lý AI từ máy tính cá nhân) và Frontend được đưa lên nền tảng đám mây.

### 1. Khởi động Backend (Máy chủ AI)
Cần có Python 3.10+ trên máy tính.

1. Mở terminal và di chuyển vào thư mục `backend`:
   ```bash
   cd backend
   ```
2. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```
3. Đảm bảo phần mềm **Ollama** đã được bật và đang chạy.
4. Chạy máy chủ FastAPI:
   ```bash
   uvicorn main:app --reload
   ```
5. Mở 1 terminal mới, dùng **Ngrok** để tạo đường dẫn Public cho API:
   ```bash
   ngrok http 8000
   ```
6. Ngrok sẽ cấp một đường link HTTPS (ví dụ: `https://abcd-123.ngrok-free.app`). **Giữ nguyên cả 2 cửa sổ terminal này.**

### 2. Triển khai Frontend (Giao diện người dùng)
1. Mở file `frontend/script.js`.
2. Sửa biến `BACKEND_URL` ở đầu file thành đường link Ngrok vừa nhận được.
3. Sửa biến `WS_URL` thành đường link websocket tương ứng của Ngrok (ví dụ: `wss://abcd-123.ngrok-free.app/ws/ar-stream`).
4. Truy cập **[Netlify Drop](https://app.netlify.com/drop)** và kéo thả thư mục `frontend` vào để lấy link trang web chính thức (ví dụ: `https://my-app.netlify.app`).

**Lưu ý:** Giao diện trên Netlify sẽ gọi dữ liệu qua mạng về máy tính của bạn thông qua đường hầm Ngrok. Vì vậy, máy tính của bạn phải luôn được bật và giữ kết nối 2 cửa sổ Terminal (Backend và Ngrok) trong suốt quá trình người dùng sử dụng ứng dụng.