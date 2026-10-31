import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
from recognizer import ISLRecognizer

BG = "#F4F8FC"
CARD = "#FFFFFF"
TEXT = "#20334D"
MUTED = "#65758B"
BLUE = "#3478D4"
GREEN = "#16865B"

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("ISL Learning Assistant")
        self.root.geometry("1180x760")
        self.root.configure(bg=BG)
        try:
            self.recognizer = ISLRecognizer()
        except Exception as e:
            messagebox.showerror("Could not load ISL model", str(e))
            root.destroy()
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera error", "Could not open webcam.")
            self.recognizer.close()
            root.destroy()
            return
        self.build_ui()
        self.running = True
        self.update_frame()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def build_ui(self):
        header = tk.Frame(self.root, bg=CARD, padx=24, pady=18)
        header.pack(fill="x", padx=18, pady=(18,12))
        tk.Label(header, text="✋  ISL Learning Assistant", bg=CARD, fg=TEXT,
                 font=("Segoe UI", 22, "bold")).pack(side="left")
        tk.Label(header, text="INDIAN SIGN LANGUAGE RECOGNITION", bg=CARD, fg=BLUE,
                 font=("Segoe UI", 10, "bold")).pack(side="right", pady=10)

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=18, pady=(0,18))
        left = tk.Frame(body, bg=CARD, padx=12, pady=12)
        left.pack(side="left", fill="both", expand=True, padx=(0,10))
        right = tk.Frame(body, bg=BG, width=330)
        right.pack(side="right", fill="y")
        tk.Label(left, text="LIVE CAMERA", bg=CARD, fg=MUTED,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0,8))
        self.video_label = tk.Label(left, bg="#E8EEF5", text="Starting camera...",
                                    fg=MUTED, font=("Segoe UI", 14))
        self.video_label.pack(fill="both", expand=True)

        result = tk.Frame(right, bg="#E7F7EF", padx=20, pady=22)
        result.pack(fill="x", pady=(0,12))
        tk.Label(result, text="RECOGNIZED SIGN", bg="#E7F7EF", fg=GREEN,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.pred_label = tk.Label(result, text="Ready", bg="#E7F7EF", fg="#126B49",
                                   font=("Segoe UI", 23, "bold"), wraplength=285)
        self.pred_label.pack(anchor="w", pady=(12,8))
        self.conf_label = tk.Label(result, text="Confidence appears after recognition",
                                   bg="#E7F7EF", fg=TEXT, font=("Segoe UI", 10))
        self.conf_label.pack(anchor="w")

        status = tk.Frame(right, bg=CARD, padx=18, pady=18)
        status.pack(fill="x", pady=(0,12))
        tk.Label(status, text="STATUS", bg=CARD, fg=MUTED,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.status_label = tk.Label(status, text="Waiting for camera…", bg=CARD, fg=TEXT,
                                     font=("Segoe UI", 11), wraplength=280, justify="left")
        self.status_label.pack(anchor="w", pady=(8,0))

        guide = tk.Frame(right, bg=CARD, padx=18, pady=18)
        guide.pack(fill="x")
        tk.Label(guide, text="HOW TO USE", bg=CARD, fg=TEXT,
                 font=("Segoe UI", 12, "bold")).pack(anchor="w")
        guide_text = ("1. Keep your hands visible.\n"
                      "2. Perform one sign at a time.\n"
                      "3. Briefly remove your hands between signs for more reliable results.")
        tk.Label(guide, text=guide_text, bg=CARD, fg=MUTED, justify="left",
                 wraplength=280, font=("Segoe UI", 10), pady=10).pack(anchor="w")
        tk.Label(self.root, text="AI-powered student prototype  •  College PBL Project",
                 bg=BG, fg=MUTED, font=("Segoe UI", 9)).pack(pady=(0,12))

    def update_frame(self):
        if not self.running:
            return
        ok, frame = self.cap.read()
        if ok:
            frame = cv2.flip(frame, 1)
            result = self.recognizer.process(frame)
            rgb = cv2.cvtColor(result["frame"], cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            image.thumbnail((760, 560))
            photo = ImageTk.PhotoImage(image)
            self.video_label.configure(image=photo, text="")
            self.video_label.image = photo
            self.pred_label.configure(text=result["prediction"].upper() if result["prediction"] else "—")
            if result["prediction"]:
                self.conf_label.configure(text=f"Confidence: {result['confidence']*100:.1f}%")
            self.status_label.configure(text=result["status"])
        self.root.after(15, self.update_frame)

    def close(self):
        self.running = False
        self.cap.release()
        self.recognizer.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
