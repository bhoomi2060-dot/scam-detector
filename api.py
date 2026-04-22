from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import mysql.connector
import hashlib

app = FastAPI()

# ============ DATABASE CONNECTION ============
def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="scam_detector"
    )

# ============ LOAD MODELS ============
folder = r"C:\Users\Admin\OneDrive\Desktop\scam_project\models"
sms_model   = joblib.load(f"{folder}/sms_model.pkl")
tfidf_sms   = joblib.load(f"{folder}/tfidf_sms.pkl")
email_model = joblib.load(f"{folder}/email_model.pkl")
tfidf_email = joblib.load(f"{folder}/tfidf_email.pkl")

# ============ MODELS ============
class TextInput(BaseModel):
    text: str

class UserRegister(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class ScanSave(BaseModel):
    user_id: int
    scan_type: str
    input_text: str
    result: str
    confidence: str

# ============ HELPER ============
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# ============ AUTH ROUTES ============
@app.post("/register")
def register(user: UserRegister):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT id FROM users WHERE email = %s", (user.email,))
    existing = cursor.fetchone()
    if existing:
        raise HTTPException(status_code=400, detail="Account already exists! Please login.")
    hashed = hash_password(user.password)
    cursor.execute(
        "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
        (user.username, user.email, hashed)
    )
    db.commit()
    user_id = cursor.lastrowid
    cursor.close()
    db.close()
    return {"message": "Account created!", "user_id": user_id, "username": user.username}

@app.post("/login")
def login(user: UserLogin):
    db = get_db()
    cursor = db.cursor()
    hashed = hash_password(user.password)
    cursor.execute(
        "SELECT id, username FROM users WHERE email = %s AND password = %s",
        (user.email, hashed)
    )
    result = cursor.fetchone()
    cursor.close()
    db.close()
    if not result:
        raise HTTPException(status_code=401, detail="Invalid email or password!")
    return {"message": "Login successful!", "user_id": result[0], "username": result[1]}

# ============ SCAN HISTORY ============
@app.post("/save_scan")
def save_scan(scan: ScanSave):
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        "INSERT INTO scan_history (user_id, scan_type, input_text, result, confidence) VALUES (%s, %s, %s, %s, %s)",
        (scan.user_id, scan.scan_type, scan.input_text, scan.result, scan.confidence)
    )
    db.commit()
    cursor.close()
    db.close()
    return {"message": "Scan saved!"}

@app.get("/scan_history/{user_id}")
def get_history(user_id: int):
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        "SELECT scan_type, input_text, result, confidence, scanned_at FROM scan_history WHERE user_id = %s ORDER BY scanned_at DESC LIMIT 20",
        (user_id,)
    )
    rows = cursor.fetchall()
    cursor.close()
    db.close()
    return {"history": [
        {"type": r[0], "text": r[1], "result": r[2], "confidence": r[3], "time": str(r[4])}
        for r in rows
    ]}

# ============ DETECTION ROUTES ============
@app.post("/check_sms")
def check_sms(data: TextInput):
    vec = tfidf_sms.transform([data.text])
    result = sms_model.predict(vec)[0]
    return {"result": result}

@app.post("/check_email")
def check_email(data: TextInput):
    vec = tfidf_email.transform([data.text])
    result = email_model.predict(vec)[0]
    return {"result": result}

@app.get("/")
def home():
    return {"message": "Scam Detector API is running!"}