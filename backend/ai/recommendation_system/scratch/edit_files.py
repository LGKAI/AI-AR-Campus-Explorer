# scratch/edit_files.py
import json

# 1. Update engine/model_metadata.json
path_meta = "engine/model_metadata.json"
try:
    with open(path_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    nodes_list = meta["crowd"]["nodes"]
    if "Căn tin" not in nodes_list:
        nodes_list.append("Căn tin")
        meta["crowd"]["nodes"] = nodes_list
        with open(path_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print("Updated NODES list in engine/model_metadata.json")
except Exception as e:
    print(f"Error updating model_metadata.json: {e}")

# 2. Update engine/campus_knowledge.py
path_ck = "engine/campus_knowledge.py"
try:
    with open(path_ck, "r", encoding="utf-8") as f:
        content_ck = f.read()

    # Let's add edge from Tòa D to Căn tin in campus knowledge graph
    if '"Căn tin"' not in content_ck:
        edge_str = '    {"from": "Căn tin", "to": "Tòa D", "relation": "co_occurs_with",\n     "tags": ["can tin", "thu vien"], "weight": 0.9},'
        content_ck = content_ck.replace("KNOWLEDGE_GRAPH_EDGES: List[dict] = [", f"KNOWLEDGE_GRAPH_EDGES: List[dict] = [\n{edge_str}")
        
        # Also add Căn tin to REVIEW_SIGNALS
        review_str = '    "Căn tin": [\n        {"phrase": "can tin an trua", "keywords": ["can tin", "an", "trua"], "time_band": "lunch"},\n    ],'
        content_ck = content_ck.replace("REVIEW_SIGNALS: Dict[str, List[dict]] = {", f"REVIEW_SIGNALS: Dict[str, List[dict]] = {{\n{review_str}")
        
        # Also update typical_hours canteen_peak
        content_ck = content_ck.replace('"Tòa D: thư viện, căn tin, quầy giao trình"', '"Tòa D: thư viện, quầy giáo trình · Căn tin: ăn uống"')
        
        with open(path_ck, "w", encoding="utf-8") as f:
            f.write(content_ck)
        print("Updated engine/campus_knowledge.py")
    else:
        print("Căn tin already exists in campus_knowledge.py")
except Exception as e:
    print(f"Error updating campus_knowledge.py: {e}")
