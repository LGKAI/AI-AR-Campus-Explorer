# engine/utils.py
import math
from datetime import datetime, time

# --- Hằng số ---
WALKING_SPEED_MPM = 80  # Tốc độ đi bộ trung bình: 80 mét/phút (~4.8 km/h)
MIN_KEYWORD_LENGTH = 3  # Độ dài tối thiểu để tránh false positive trong NLP


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Tính khoảng cách thực tế (mét) giữa 2 điểm GPS theo công thức Haversine.
    Hàm dùng chung cho toàn bộ project — không định nghĩa lại ở nơi khác.
    """
    R = 6_371_000  # Bán kính Trái Đất tính bằng mét
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def parse_time(t_str: str) -> time:
    """Chuyển chuỗi 'HH:MM' sang đối tượng datetime.time để so sánh an toàn."""
    try:
        h, m = map(int, t_str.split(":"))
        return time(h, m)
    except (ValueError, AttributeError):
        return None


def get_current_time_str() -> str:
    """Lấy giờ hệ thống hiện tại dạng 'HH:MM'."""
    return datetime.now().strftime("%H:%M")
