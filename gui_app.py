import tkinter as tk
from tkinter import messagebox
import joblib

# ============ LOAD MODELS ============
print("Loading models...")
folder = r"C:\Users\Admin\OneDrive\Desktop\scam_project\models"

sms_model   = joblib.load(f"{folder}/sms_model.pkl")
print("SMS model loaded!")
tfidf_sms   = joblib.load(f"{folder}/tfidf_sms.pkl")
print("SMS tfidf loaded!")
email_model = joblib.load(f"{folder}/email_model.pkl")
print("Email model loaded!")
tfidf_email = joblib.load(f"{folder}/tfidf_email.pkl")
print("Email tfidf loaded!")
print("All models loaded! Opening GUI...")

# ============ PREDICT FUNCTIONS ============
def predict_sms(text):
    vec = tfidf_sms.transform([text])
    return sms_model.predict(vec)[0]

def predict_email(text):
    vec = tfidf_email.transform([text])
    return email_model.predict(vec)[0]

# ============ GUI ============
root = tk.Tk()
root.title("Scam Detector")
root.geometry("600x500")
root.configure(bg="#1e1e2e")
root.resizable(False, False)

# --- Title ---
tk.Label(root, text="SCAM DETECTOR", font=("Arial", 22, "bold"),
         bg="#1e1e2e", fg="#cdd6f4").pack(pady=20)

# --- Input Box ---
tk.Label(root, text="Enter SMS / Email text below:",
         font=("Arial", 11), bg="#1e1e2e", fg="#a6adc8").pack()

text_box = tk.Text(root, height=8, width=60, font=("Arial", 11),
                   bg="#313244", fg="#cdd6f4", insertbackground="white",
                   relief="flat", padx=10, pady=10)
text_box.pack(pady=10)

# --- Result Label ---
result_var = tk.StringVar()
result_var.set("Result will appear here...")
result_label = tk.Label(root, textvariable=result_var,
                        font=("Arial", 14, "bold"),
                        bg="#1e1e2e", fg="#a6e3a1")
result_label.pack(pady=10)

# --- Button Functions ---
def check_sms():
    text = text_box.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Warning", "Please enter some text!")
        return
    pred = predict_sms(text)
    if pred == "ham":
        result_var.set("✅ SMS is SAFE (Ham)")
        result_label.config(fg="#a6e3a1")
    elif pred == "spam":
        result_var.set("❌ SMS is SPAM!")
        result_label.config(fg="#f38ba8")
    else:
        result_var.set("⚠️ SMS is SMISHING (Scam)!")
        result_label.config(fg="#fab387")

def check_email():
    text = text_box.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Warning", "Please enter some text!")
        return
    pred = predict_email(text)
    if pred == "ham":
        result_var.set("✅ Email is SAFE")
        result_label.config(fg="#a6e3a1")
    else:
        result_var.set("❌ Email is SPAM/SCAM!")
        result_label.config(fg="#f38ba8")

def clear_all():
    text_box.delete("1.0", tk.END)
    result_var.set("Result will appear here...")
    result_label.config(fg="#a6e3a1")

# --- Buttons ---
btn_frame = tk.Frame(root, bg="#1e1e2e")
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="Check SMS", font=("Arial", 12, "bold"),
          bg="#89b4fa", fg="#1e1e2e", padx=20, pady=8,
          relief="flat", command=check_sms).grid(row=0, column=0, padx=10)

tk.Button(btn_frame, text="Check Email", font=("Arial", 12, "bold"),
          bg="#a6e3a1", fg="#1e1e2e", padx=20, pady=8,
          relief="flat", command=check_email).grid(row=0, column=1, padx=10)

tk.Button(btn_frame, text="Clear", font=("Arial", 12, "bold"),
          bg="#f38ba8", fg="#1e1e2e", padx=20, pady=8,
          relief="flat", command=clear_all).grid(row=0, column=2, padx=10)

# --- Footer ---
tk.Label(root, text="Scam Detector | Random Forest Model",
         font=("Arial", 9), bg="#1e1e2e", fg="#585b70").pack(side="bottom", pady=10)

root.mainloop()