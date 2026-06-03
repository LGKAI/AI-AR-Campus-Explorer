from fastapi import APIRouter, HTTPException, Depends, status
from core.config import db
from core.security import (
    get_password_hash, verify_password, create_access_token, 
    oauth2_scheme, SECRET_KEY, ALGORITHM
)
from models.schemas import UserCreate, UserLogin, UserResponse, Token
from ai.face_guard.adapter import register_face, verify_face
from jose import JWTError, jwt
from datetime import timedelta
import uuid

router = APIRouter(prefix="/users", tags=["Users"])

@router.post("/", response_model=UserResponse)
async def create_user(user: UserCreate):
    users_ref = db.collection("USER")
    
    # Kiểm tra trùng lặp tài khoản dựa trên Email CSDL
    query = users_ref.where("Email", "==", user.Email).get()
    if query:
        raise HTTPException(status_code=400, detail="Email đã được đăng ký")
    
    user_id = str(uuid.uuid4())
    
    # Bước băm khuôn mặt và đẩy lưu trữ vào tập tin chỉ mục FAISS
    is_face_saved = register_face(user.FaceData, user_id)
    if not is_face_saved:
        raise HTTPException(
            status_code=400, 
            detail="Không trích xuất được vector khuôn mặt. Hãy giữ góc nhìn thẳng và đủ sáng."
        )
    
    # Khởi tạo bản ghi lưu trữ Firebase
    user_data = {
        "UserID": user_id,
        "FullName": user.FullName,
        "Email": user.Email,
        "PasswordHash": get_password_hash(user.Password)
    }
    users_ref.document(user_id).set(user_data)
    return UserResponse(**user_data)

@router.post("/login", response_model=Token)
async def login(user: UserLogin):
    users_ref = db.collection("USER")
    query = users_ref.where("Email", "==", user.Email).get()
    
    if not query:
        raise HTTPException(status_code=400, detail="Tài khoản email không tồn tại trong hệ thống")
    
    user_doc = query[0].to_dict()
    is_authenticated = False
    
    # Nhánh 1: Sử dụng chuỗi mật khẩu thông thường
    if user.Password and len(user.Password.strip()) > 0:
        if verify_password(user.Password, user_doc["PasswordHash"]):
            is_authenticated = True
        else:
            raise HTTPException(status_code=400, detail="Mật khẩu xác thực không chính xác")
            
    # Nhánh 2: Xác thực thông qua Face Guard tích hợp FAISS
    elif user.FaceData and len(user.FaceData.strip()) > 0:
        recognized_id = verify_face(user.FaceData)
        if recognized_id == "NO_FACE":
            raise HTTPException(status_code=400, detail="Không phát hiện được cấu trúc khuôn mặt trong khung hình")
        elif recognized_id == "UNKNOWN" or recognized_id != user_doc["UserID"]:
            raise HTTPException(status_code=403, detail="Xác thực FaceID thất bại. Khuôn mặt không khớp")
        else:
            is_authenticated = True
            
    # Trường hợp thiếu cả hai trường
    else:
        raise HTTPException(
            status_code=400, 
            detail="Yêu cầu phương thức xác thực: Vui lòng nhập mật khẩu hoặc quét khuôn mặt"
        )
    
    # Cấp phát chuỗi JWT Token định danh nếu vượt qua bộ lọc xác thực thành công
    if is_authenticated:
        access_token = create_access_token(
            data={"sub": user_doc["Email"]}, 
            expires_delta=timedelta(minutes=60)
        )
        return {"access_token": access_token, "token_type": "bearer"}
    
    raise HTTPException(status_code=401, detail="Xác thực hệ thống không hợp lệ")

# Định nghĩa dependency chuẩn của FastAPI
oauth2_dependency = Depends(oauth2_scheme)

async def get_current_user(token: str = oauth2_dependency):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Không thể xác thực thông tin (Token không hợp lệ)",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    query = db.collection("USER").where("Email", "==", email).get()
    if not query:
        raise credentials_exception
    return query[0].to_dict()

@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return UserResponse(**current_user)