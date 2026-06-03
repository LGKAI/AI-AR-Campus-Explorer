from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class UserCreate(BaseModel):
    FullName: str
    Email: EmailStr
    Password: str = Field(..., min_length=6) # Độ dài tối thiểu của mật khẩu là 6 ký tự
    FaceData: str  # Dữ liệu hình ảnh base64 gửi lên để face_guard xử lý

class UserLogin(BaseModel):
    Email: EmailStr
    Password: str
    FaceData: Optional[str] = None  # Dữ liệu hình ảnh base64 gửi lên để face_guard xử lý

class UserResponse(BaseModel):
    UserID: str
    FullName: str
    Email: EmailStr

class Token(BaseModel):
    access_token: str
    token_type: str

class Location(BaseModel):
    LocationID: str
    LocationName: str
    Type: str
    Latitude: float
    Longitude: float
    IsARActive: bool