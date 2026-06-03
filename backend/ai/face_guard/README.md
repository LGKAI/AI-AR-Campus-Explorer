# 🎯 FaceGuard

Simple and lightweight face recognition system using:

* InsightFace (face embedding)
* FAISS (database)
* OpenCV (camera)

---

# 🚀 Setup

## 1. Create environment

```bash
conda create -n face python=3.10 -y
conda activate face
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🎥 Usage

## 🔹 Register face

```bash
python enroll.py
```

* Enter your name
* Press **S** to capture
* See: `Saved <name>`

---

## 🔹 Run recognition

```bash
python main.py
```

* System checks every 4 seconds
* Output example:

```
[RESULT] Khanh | distance = 0.52
```

or

```
[RESULT] UNKNOWN | distance = 1.40
```

---

# ⚠️ Notes

* Delete `face_db.pkl` if you change model/code
* Good lighting improves accuracy
* Register multiple samples for better results

---

# 📌 Summary

* No server needed
* Runs on CPU
* Fast & easy to use
