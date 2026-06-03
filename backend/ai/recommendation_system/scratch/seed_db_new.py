import sys
import os
import random
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import engine.storage as storage
from engine.persona_manager import compute_node_weight_score

# Tiện ích tạo thời gian rải rác trong vòng 48 giờ qua
def random_past_timestamp():
    now = time.time()
    return now - random.uniform(0, 47 * 3600)

def generate_profile_for_persona(persona_type: str) -> dict:
    role = "student"
    study_style = "silent"
    interests = ["hoc_tap"]
    schedule_class = "Tòa B"
    
    if persona_type == "student_it":
        role = "student"
        study_style = "silent"
        interests = ["cntt", "hoc_tap"]
        schedule_class = random.choice(["Tòa B", "Tòa C"])
        visits_config = {
            "Tòa C": {"v": (7, 15), "d": (120, 240), "i_pct": 0.9},
            "Tòa B": {"v": (5, 10), "d": (90, 180), "i_pct": 0.8},
            "Tòa D": {"v": (4, 8), "d": (30, 60), "i_pct": 0.7},
            "Tòa E": {"v": (1, 4), "d": (30, 65), "i_pct": 0.5},
        }
    elif persona_type == "student_general":
        role = "student"
        study_style = "group"
        interests = ["the_thao", "an_uong", "hoc_tap"]
        schedule_class = random.choice(["Tòa B", "Căn tin", "Tòa G"])
        visits_config = {
            "Nhà thể dục": {"v": (6, 12), "d": (60, 120), "i_pct": 0.95},
            "Căn tin": {"v": (5, 10), "d": (30, 60), "i_pct": 0.8},
            "Tòa G": {"v": (3, 6), "d": (45, 90), "i_pct": 0.75},
            "Tòa B": {"v": (2, 5), "d": (45, 120), "i_pct": 0.6},
        }
    elif persona_type == "lecturer":
        role = "lecturer"
        study_style = "silent"
        interests = ["hoc_tap"]
        schedule_class = "Nhà điều hành"
        visits_config = {
            "Nhà điều hành": {"v": (8, 18), "d": (180, 360), "i_pct": 0.95},
            "Tòa A": {"v": (4, 8), "d": (90, 180), "i_pct": 0.9},
            "Tòa C": {"v": (2, 5), "d": (60, 120), "i_pct": 0.8},
        }
    else: # visitor
        role = "visitor"
        study_style = "group"
        interests = ["an_uong"]
        schedule_class = "Cổng trường"
        visits_config = {
            "Cổng trường": {"v": (1, 3), "d": (5, 15), "i_pct": 1.0},
            "Nhà xe": {"v": (1, 3), "d": (10, 25), "i_pct": 1.0},
            "ATM": {"v": (1, 2), "d": (3, 10), "i_pct": 1.0},
        }

    behavior_log = {}
    visited_history = {}
    
    for node, conf in visits_config.items():
        v_count = random.randint(*conf["v"])
        dwell_total = sum(random.uniform(*conf["d"]) for _ in range(v_count))
        avg_dwell = dwell_total / v_count
        
        i_visits = int(v_count * conf["i_pct"])
        if i_visits == 0 and v_count > 0:
            i_visits = 1
            
        transient = v_count - i_visits
        
        weight = compute_node_weight_score(
            total_visits=v_count,
            avg_dwell_time_mins=avg_dwell,
            intentional_visits=i_visits
        )
        
        behavior_log[node] = {
            "total_visits": v_count,
            "total_dwell_time_mins": round(dwell_total, 1),
            "intentional_visits": i_visits,
            "transient_passes": transient,
            "weight_score": weight,
            "last_visited": time.time() - random.uniform(0, 24 * 3600)
        }
        visited_history[node] = round(weight * 100, 1)

    return {
        "role": role,
        "study_style": study_style,
        "interests": interests,
        "visited_history": visited_history,
        "schedule_class": schedule_class,
        "behavior_log": behavior_log,
        "search_history": [],
        "_last_tracking": {}
    }

def seed_database(num_profiles=120):
    print(f"🌱 Bắt đầu nạp {num_profiles} profiles (Cập nhật Căn tin) vào SQLite...")
    storage.init_db()
    
    personas_distribution = ["student_it", "student_general", "lecturer", "visitor"]
    weights = [0.4, 0.4, 0.1, 0.1]
    
    with storage._connect() as conn:
        conn.execute("DELETE FROM user_profiles WHERE session_id LIKE 'session_seed_%'")
        print("🧹 Đã dọn dẹp các session_seed cũ.")
    
    for i in range(1, num_profiles + 1):
        session_id = f"session_seed_{i:03d}"
        persona = random.choices(personas_distribution, weights=weights, k=1)[0]
        profile = generate_profile_for_persona(persona)
        
        now = random_past_timestamp()
        
        with storage._connect() as conn:
            import json
            conn.execute(
                """
                INSERT INTO user_profiles (session_id, profile_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    updated_at   = excluded.updated_at
                """,
                (session_id, json.dumps(profile, ensure_ascii=False), now),
            )
            
    print(f"🚀 Nạp thành công {num_profiles} profiles mồi.")

if __name__ == "__main__":
    seed_database(120)
