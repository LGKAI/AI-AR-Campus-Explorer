# AI AR Campus Explorer - Recommendation System Backend

Đây là hệ thống Backend cung cấp API cho ứng dụng AI AR Campus Explorer. Hệ thống sử dụng FastAPI, xử lý các tác vụ như tìm đường đi (Pathfinding) với A*, GNN-GAT, theo dõi vị trí thời gian thực, và đề xuất địa điểm thông minh dựa trên NLP, Collaborative Filtering (CF).

## 🛠 Cài đặt (Installation)

Yêu cầu hệ thống:
- Python 3.8+ (khuyến nghị)

**Bước 1:** Clone hoặc tải mã nguồn về máy.

**Bước 2:** Di chuyển vào thư mục dự án:
```bash
cd source/backend/ai/recommendation_system
```

**Bước 3:** Cài đặt các thư viện phụ thuộc:
```bash
pip install -r requirements.txt
```
*(Các thư viện chính bao gồm FastAPI, Uvicorn, NetworkX, PyTorch, PyTorch Geometric, Pydantic, v.v.)*

## 🚀 Chạy ứng dụng

Chạy server FastAPI bằng Uvicorn:
```bash
uvicorn api.main:app --reload
```
API sẽ có sẵn tại: `http://localhost:8000`
Swagger UI để test các endpoint: `http://localhost:8000/docs`

## 📂 Cấu trúc thư mục và Vai trò các file

```text
recommendation_system/
│
├── api/                   # Chứa mã nguồn cho API endpoint
│   ├── main.py            # File chạy chính của server FastAPI (entry point), định nghĩa các routes cho tìm đường, tracking, gợi ý địa điểm.
│   └── templates/         # Thư mục chứa các mẫu giao diện (nếu có dùng cho dashboard/view nhỏ)
│
├── engine/                # Core Logic của Hệ thống AI (AI Engine)
│   ├── building_catalog.py      # Quản lý hồ sơ chi tiết (sự kiện, phòng ban, dịch vụ) của các tòa nhà.
│   ├── campus_knowledge.py      # Chứa tri thức cơ sở về khuôn viên trường.
│   ├── collaborative_filter.py  # Hệ thống gợi ý Item-Item Collaborative Filtering dựa trên lịch sử truy cập.
│   ├── context_engine.py        # Động cơ xử lý ngữ cảnh (thời gian, thời tiết, sự kiện).
│   ├── context_features.py      # Trích xuất và biểu diễn đặc trưng ngữ cảnh.
│   ├── gnn_engine.py            # Động cơ mạng nơ-ron đồ thị (GNN - Graph Neural Network) cho việc nhúng node.
│   ├── graph_builder_v2.py      # Xây dựng đồ thị đường đi khuôn viên trường bằng NetworkX.
│   ├── nlp_processor.py         # Xử lý ngôn ngữ tự nhiên (Semantic matching, TF-IDF).
│   ├── optimizer.py             # Tối ưu hóa thuật toán tìm đường (A*, đa điểm).
│   ├── persona_manager.py       # Quản lý hồ sơ và cập nhật thông tin cá nhân hóa thụ động.
│   ├── recommender.py           # Engine tổng hợp đề xuất địa điểm kết hợp nhiều mô hình.
│   ├── storage.py               # Thao tác với cơ sở dữ liệu SQLite (lưu profile, lịch sử, comments).
│   ├── utils.py                 # Các hàm tiện ích chung (tính khoảng cách haversine, thời gian).
│   └── *.pth, *.json, *.db      # Các file lưu mô hình PyTorch (crowd_model, intent_model) và database SQLite.
│
├── static/                # Chứa các tài nguyên tĩnh phục vụ cho API
│   └── *.jpg              # Hình ảnh của các tòa nhà trong campus (Tòa A, B, C, Căn tin, v.v.)
│
├── requirements.txt       # Danh sách các thư viện Python cần thiết
└── README.md              # File tài liệu hướng dẫn (chính là file này)
```

## 🌟 Các tính năng chính

- **Smart Pathfinding (A* + GNN-GAT)**: Tìm đường đi ngắn nhất, có tùy chọn tránh mưa, hỗ trợ xe lăn, tính khoảng cách, ước tính thời gian.
- **Geofencing & Tracking**: Theo dõi người dùng thời gian thực, bật cảnh báo khi vào vùng cấm hoặc khi đến nơi.
- **Smart Recommendations (Cá nhân hóa + AI)**: Đề xuất địa điểm dựa vào vị trí, thời tiết, ngữ cảnh (giờ giấc, pin) và sở thích cá nhân. Kết hợp Collaborative Filtering (CF).
- **SQLite Persistent Storage**: Lưu trữ user profile, rating, và comment lâu dài, không bị mất khi khởi động lại server. Mở rộng khả năng gợi ý thụ động dựa trên lịch sử.
