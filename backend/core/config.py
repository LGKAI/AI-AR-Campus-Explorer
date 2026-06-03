import firebase_admin
from firebase_admin import credentials, firestore
import os

# Đường dẫn tĩnh tới serviceAccountKey.json nằm cùng thư mục
cred_path = os.path.join(os.path.dirname(__file__), "serviceAccountKey.json")

if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)

db = firestore.client()