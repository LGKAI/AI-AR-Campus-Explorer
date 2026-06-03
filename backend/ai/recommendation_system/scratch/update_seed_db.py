# scratch/update_seed_db.py

path_seed = "engine/seed_db.py"
with open(path_seed, "r", encoding="utf-8") as f:
    code = f.read()

# Replace Tòa D with Căn tin in visits_config for student_general
# And adjust visitor/other configs if needed
code = code.replace('"Tòa D": {"v": (5, 10), "d": (30, 60), "i_pct": 0.8}', '"Căn tin": {"v": (5, 10), "d": (30, 60), "i_pct": 0.8}')
code = code.replace('interests = ["the_thao", "an_uong", "hoc_tap"]\n        schedule_class = random.choice(["Tòa B", "Tòa D", "Tòa G"])', 'interests = ["the_thao", "an_uong", "hoc_tap"]\n        schedule_class = random.choice(["Tòa B", "Căn tin", "Tòa G"])')
code = code.replace('food_w    = _w("Tòa D")', 'food_w    = _w("Căn tin")')

with open(path_seed, "w", encoding="utf-8") as f:
    f.write(code)

print("Updated engine/seed_db.py successfully!")
