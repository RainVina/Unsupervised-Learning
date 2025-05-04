import cv2
import face_recognition
import numpy as np
from sklearn.cluster import DBSCAN
import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import csv
from datetime import datetime
from collections import defaultdict
import time

# Create output directory and log file
OUTPUT_DIR = "frequent_customers"
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUTPUT_DIR, "visit_log.csv")

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Label", "Timestamp"])

# Parameters
ENCODINGS = []
FACES = []
label_count = {}
dbscan = DBSCAN(eps=0.5, min_samples=3)
last_visit_time = {}  # To store last visit time for each label
COOLDOWN_TIME = 30  # Default cooldown in seconds

# GUI Application Class
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Frequent Customer Detector")

        self.running = False
        self.cap = None
        self.thread = None

        self.video_label = tk.Label(root)
        self.video_label.pack(padx=10, pady=10)

        # Button panel
        btn_frame = ttk.Frame(root, padding=10)
        btn_frame.pack(fill=tk.X)

        ttk.Button(btn_frame, text="Start Camera", command=self.start_camera).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Stop Camera", command=self.stop_camera).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Show Frequent Customers", command=self.show_customers).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Exit", command=self.root.quit).pack(side=tk.RIGHT, padx=5)

        # Cooldown control
        self.cooldown_label = ttk.Label(root, text="Cooldown (seconds):")
        self.cooldown_label.pack(pady=5)
        self.cooldown_slider = ttk.Scale(root, from_=10, to_=60, orient="horizontal", length=300)
        self.cooldown_slider.set(COOLDOWN_TIME)
        self.cooldown_slider.pack(pady=5)

        self.cooldown_checkbox_var = tk.BooleanVar(value=True)
        self.cooldown_checkbox = ttk.Checkbutton(
            root, text="Enable Cooldown", variable=self.cooldown_checkbox_var
        )
        self.cooldown_checkbox.pack(pady=5)

    def start_camera(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.thread = threading.Thread(target=self.video_loop)
        self.thread.start()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def log_visit(self, label):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([label, timestamp])

    def video_loop(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for enc, loc in zip(encodings, face_locations):
                ENCODINGS.append(enc)
                FACES.append((frame.copy(), loc))

                # If the cooldown is enabled
                if self.cooldown_checkbox_var.get():
                    current_time = time.time()
                    label = dbscan.fit_predict([enc])  # Using DBSCAN to assign labels
                    if label == -1:
                        continue
                    
                    if label not in last_visit_time or (current_time - last_visit_time[label]) >= self.cooldown_slider.get():
                        last_visit_time[label] = current_time
                        self.save_image(label, FACES[-1])

            for top, right, bottom, left in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        if self.cap:
            self.cap.release()

    def save_image(self, label, face_data):
        img, loc = face_data
        top, right, bottom, left = loc
        crop = img[top:bottom, left:right]
        existing = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(f"customer_{label}_")]
        next_index = len(existing)
        folder = os.path.join(OUTPUT_DIR, f"customer_{label}")
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, f"visit_{next_index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(filename, crop)
        self.log_visit(label)

    def show_customers(self):
        files = os.listdir(OUTPUT_DIR)
        if not files:
            messagebox.showinfo("Info", "No frequent customers saved yet.")
            return

        popup = tk.Toplevel(self.root)
        popup.title("Frequent Customers")
        popup.geometry("600x500")

        canvas = tk.Canvas(popup)
        scrollbar = ttk.Scrollbar(popup, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        customer_data = defaultdict(list)
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    label = row["Label"]
                    timestamp = row["Timestamp"]
                    customer_data[label].append(timestamp)

        for label in sorted(customer_data.keys(), key=lambda x: int(x)):
            img_path = None
            for file in os.listdir(OUTPUT_DIR):
                if file.startswith(f"customer_{label}_"):
                    img_path = os.path.join(OUTPUT_DIR, file)
                    break
            if not img_path:
                continue

            img = Image.open(img_path).resize((100, 100))
            imgtk = ImageTk.PhotoImage(img)

            frame = ttk.Frame(scroll_frame, padding=10)
            ttk.Label(frame, image=imgtk).pack()
            ttk.Label(frame, text=f"Customer #{label}").pack()
            ttk.Label(frame, text=f"Visits: {len(customer_data[label])}").pack()
            ttk.Label(frame, text=f"Last seen: {customer_data[label][-1]}").pack()
            frame.image = imgtk
            frame.pack(side=tk.TOP, anchor="w", pady=10)

# Launch
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
