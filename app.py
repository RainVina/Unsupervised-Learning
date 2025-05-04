import cv2
import face_recognition
import numpy as np
from sklearn.cluster import DBSCAN
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import threading
import csv
from datetime import datetime
from collections import defaultdict
import shutil

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

        self.cooldown_seconds = tk.IntVar(value=30)  # Default cooldown
        self.last_capture_time = {}  # Per label cooldown tracking

        # Button panel
        btn_frame = ttk.Frame(root, padding=10)
        btn_frame.pack(fill=tk.X)

        ttk.Button(btn_frame, text="Start Camera", command=self.start_camera).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Stop Camera", command=self.stop_camera).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Show Frequent Customers", command=self.show_customers).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Export Report", command=self.export_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Reset Data", command=self.reset_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Exit", command=self.root.quit).pack(side=tk.RIGHT, padx=5)

        # Cooldown setting
        cooldown_frame = ttk.Frame(root, padding=5)
        cooldown_frame.pack()
        ttk.Label(cooldown_frame, text="Capture Cooldown (seconds):").pack(side=tk.LEFT)
        cooldown_menu = ttk.OptionMenu(cooldown_frame, self.cooldown_seconds, 30, 30, 15, 10, 5)
        cooldown_menu.pack(side=tk.LEFT)

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

                if len(ENCODINGS) % 10 == 0:
                    labels = dbscan.fit_predict(ENCODINGS)
                    for i, label in enumerate(labels):
                        if label == -1:
                            continue
                        label_count[label] = label_count.get(label, 0) + 1

                        now = datetime.now()
                        last_time = self.last_capture_time.get(label)
                        cooldown = self.cooldown_seconds.get()

                        if not last_time or (now - last_time).total_seconds() >= cooldown:
                            img, loc = FACES[i]
                            top, right, bottom, left = loc
                            crop = img[top:bottom, left:right]
                            customer_dir = os.path.join(OUTPUT_DIR, f"customer_{label}")
                            os.makedirs(customer_dir, exist_ok=True)
                            timestamp = now.strftime("%Y%m%d_%H%M%S")
                            filename = os.path.join(customer_dir, f"visit_{label}_{timestamp}.jpg")
                            cv2.imwrite(filename, crop)
                            self.log_visit(label)
                            self.last_capture_time[label] = now

            for top, right, bottom, left in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        if self.cap:
            self.cap.release()

    def show_customers(self):
        if not os.path.exists(LOG_FILE):
            messagebox.showinfo("Info", "No visit log available.")
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
        with open(LOG_FILE, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row["Label"]
                timestamp = row["Timestamp"]
                customer_data[label].append(timestamp)

        for label in sorted(customer_data.keys(), key=lambda x: int(x)):
            customer_dir = os.path.join(OUTPUT_DIR, f"customer_{label}")
            if not os.path.exists(customer_dir):
                continue
            images = [f for f in os.listdir(customer_dir) if f.endswith(".jpg")]
            if not images:
                continue
            img_path = os.path.join(customer_dir, images[0])
            img = Image.open(img_path).resize((100, 100))
            imgtk = ImageTk.PhotoImage(img)

            frame = ttk.Frame(scroll_frame, padding=10)
            ttk.Label(frame, image=imgtk).pack()
            ttk.Label(frame, text=f"Customer #{label}").pack()
            ttk.Label(frame, text=f"Visits: {len(customer_data[label])}").pack()
            ttk.Label(frame, text=f"Last seen: {customer_data[label][-1]}").pack()
            frame.image = imgtk
            frame.pack(side=tk.TOP, anchor="w", pady=10)

    def export_report(self):
        if not os.path.exists(LOG_FILE):
            messagebox.showinfo("Info", "No visit log to export.")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if save_path:
            shutil.copy(LOG_FILE, save_path)
            messagebox.showinfo("Success", f"Report saved to {save_path}")

    def reset_data(self):
        if messagebox.askyesno("Confirm", "Are you sure you want to delete all saved data?"):
            shutil.rmtree(OUTPUT_DIR)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(LOG_FILE, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Label", "Timestamp"])
            global ENCODINGS, FACES, label_count
            ENCODINGS = []
            FACES = []
            label_count = {}
            self.last_capture_time.clear()
            messagebox.showinfo("Reset", "All data has been reset.")

# Launch
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
