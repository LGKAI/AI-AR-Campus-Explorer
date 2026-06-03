# scratch/test_query.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.graph_builder_v2 import build_campus_graph
from engine.recommender import recommend_locations, get_smart_recommendations

def main():
    G = build_campus_graph()
    
    queries = ["ăn trưa", "tự học", "thể thao", "ăn uống", "tự học yên tĩnh"]
    for q in queries:
        print(f"\n=== Query: {q} (recommend_locations) ===")
        recs = recommend_locations(G, query=q, current_time="12:00", limit=5)
        for i, r in enumerate(recs):
            print(f"{i+1}. {r['node']} - Score: {r['score']} - Reason: {r['reason']}")
            
    print("\n=== Smart Recommendations (Query: ăn uống) ===")
    smart_recs = get_smart_recommendations(
        G,
        current_lat=10.877500,
        current_lon=106.798000,  # near Tòa B
        query="ăn uống",
        current_time_str="12:00",
        user_interests=["an_uong"]
    )
    for i, r in enumerate(smart_recs):
        print(f"{i+1}. {r['node']} - Score: {r['score']} (Source: {r.get('source')}, Bucket: {r.get('bucket')})")
        print(f"   Reason: {r['reason']}")

if __name__ == "__main__":
    main()
