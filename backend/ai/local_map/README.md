# Local Map AR WebXR Module

Đây là module Backend cung cấp dữ liệu cho tính năng AR WebXR Camera Navigation.

## Cấu trúc:
- `data/buildings.json`: Lưu trữ thông tin chi tiết các toà nhà (tên, toạ độ, icon, danh sách POI).
- `data/nav_graph.json`: Đồ thị điều hướng (các điểm node và khoảng cách) phục vụ cho thuật toán tìm đường (Dijkstra) trên AR Camera.
- `router.py`: FastAPI Router cung cấp API truy xuất và cập nhật dữ liệu.

## Các API:
- `GET /local_map/api/buildings`: Lấy danh sách toà nhà.
- `POST /local_map/api/buildings`: Cập nhật danh sách toà nhà.
- `GET /local_map/api/graph`: Lấy đồ thị điều hướng (nodes & edges).
