"""
face_recognition_pro.py
========================
Tkinter GUI  —  Face Recognition System  v3

NEW in v3
  ✔ Add Person Wizard  — create a new identity directly from the UI
      • 📸 Capture from camera  (auto-captures N frames with countdown)
      • 📁 Import from folder   (file-dialog, any image format)
  ✔ Hot-reload  — dataset panel + model refresh automatically after adding
  ✔ Delete Person  — right-click any row in the dataset panel

EXISTING
  ✔ Tkinter GUI with embedded live camera feed
  ✔ Forced image processing — 5-pass dual-cascade + centre-crop fallback
  ✔ Normal camera (no mirror flip)
  ✔ CLAHE + sharpening quality enhancement on the live feed
  ✔ Smart training — retrain only when SHA-256 fingerprint changes
  ✔ Opens https://github.com/thelastx7 on first face match (boolean guard)
"""

import cv2, os, sys, time, hashlib, json, shutil, webbrowser, threading
import numpy as np
from pathlib import Path
from typing import Optional
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from PIL import Image, ImageTk

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

TRAINING_DATA_DIR    = "training-data"
CASCADE_LBP          = "opencv-files/lbpcascade_frontalface.xml"
CASCADE_HAAR         = "opencv-files/haarcascade_frontalface_alt.xml"
MODEL_PATH           = "modele_visages.yml"
LABELS_PATH          = "modele_visages_labels.json"
FINGERPRINT_PATH     = "training_fingerprint.json"

CONFIDENCE_THRESHOLD = 85.0    # LBPH distance: lower=stricter. 85 suits real-webcam use.
WEBCAM_INDEX         = 0
CAMERA_W             = 800
CAMERA_H             = 600
CAPTURE_SAMPLES      = 40          # frames captured per new person
CAPTURE_INTERVAL_MS  = 300         # ms between auto-captures
MATCH_URL            = "https://github.com/thelastx7"
IMAGE_EXTS           = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Palette ─────────────────────────────────────────────────────────────
BG_DARK   = "#0d1117"
BG_MID    = "#161b22"
BG_CARD   = "#21262d"
BG_INPUT  = "#2d333b"
ACCENT    = "#58a6ff"
ACCENT2   = "#3fb950"
ACCENT3   = "#d29922"
WARN      = "#f85149"
TEXT      = "#e6edf3"
TEXT_DIM  = "#8b949e"
BAR_FILL  = "#238636"
BAR_EMPTY = "#2d333b"

FT       = ("Segoe UI", 10)
FT_BOLD  = ("Segoe UI", 10, "bold")
FT_TITLE = ("Segoe UI", 12, "bold")
FT_SMALL = ("Segoe UI", 9)
FT_MONO  = ("Consolas", 10)


# ═══════════════════════════════════════════════════════════════════════
#  QUALITY HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _gamma_correct(frame: np.ndarray, gamma: float = 1.4) -> np.ndarray:
    """Brighten dark frames with a gamma look-up table (fast, per-pixel)."""
    inv = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv) * 255
                    for i in range(256)], dtype=np.uint8)
    return cv2.LUT(frame, lut)


def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """
    Full quality pipeline for the live display:
      1. Gamma correction  → lifts shadows on dark cameras
      2. CLAHE in LAB space → boosts local contrast without colour shift
      3. Gentle unsharp mask → sharpens edges
    NO cv2.flip → camera stays in normal (non-mirrored) orientation.
    """
    bright = _gamma_correct(frame, gamma=1.35)
    lab    = cv2.cvtColor(bright, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe  = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    out    = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
    k = np.array([[0, -0.35, 0], [-0.35, 2.4, -0.35], [0, -0.35, 0]], np.float32)
    return cv2.filter2D(out, -1, k)


def preprocess_gray(gray: np.ndarray) -> np.ndarray:
    """Histogram equalisation + CLAHE + bilateral denoise for detection/recognition."""
    eq    = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    return cv2.bilateralFilter(clahe.apply(eq), 7, 50, 50)


# ═══════════════════════════════════════════════════════════════════════
#  CASCADES
# ═══════════════════════════════════════════════════════════════════════

def load_cascades():
    def _load(p):
        if not os.path.exists(p): return None
        cc = cv2.CascadeClassifier(p)
        return cc if not cc.empty() else None
    return _load(CASCADE_LBP), _load(CASCADE_HAAR)


# ═══════════════════════════════════════════════════════════════════════
#  FORCED FACE DETECTION  (5-pass)
# ═══════════════════════════════════════════════════════════════════════

def detect_face_forced(gray, lbp, haar) -> Optional[np.ndarray]:
    """
    5-pass strategy — returns 100×100 grayscale ROI or None.
    Pass 1-2  LBP  (strict → permissive)
    Pass 3-4  Haar (strict → permissive)
    Pass 5    Centre-crop fallback (always succeeds on non-empty images)
    """
    enh = preprocess_gray(gray)
    passes = []
    if lbp:
        passes += [(lbp,  1.10, 3, (20, 20)), (lbp,  1.05, 2, (15, 15))]
    if haar:
        passes += [(haar, 1.10, 3, (20, 20)), (haar, 1.05, 1, (15, 15))]

    for casc, sc, nb, ms in passes:
        for img in (enh, gray):
            fs = casc.detectMultiScale(img, scaleFactor=sc,
                                       minNeighbors=nb, minSize=ms)
            if len(fs):
                x, y, w, h = max(fs, key=lambda f: f[2]*f[3])
                return cv2.resize(enh[y:y+h, x:x+w], (100, 100))

    # Fallback: centre 60 %
    H, W = gray.shape
    my, mx = int(H*.2), int(W*.2)
    crop = gray[my:H-my, mx:W-mx]
    return cv2.resize(preprocess_gray(crop), (100, 100)) if crop.size else None


# ═══════════════════════════════════════════════════════════════════════
#  DATASET SCAN
# ═══════════════════════════════════════════════════════════════════════

def scan_dataset(d: str) -> list:
    root = Path(d)
    if not root.is_dir(): return []
    result = []
    for folder in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not folder.is_dir(): continue
        count = sum(1 for f in folder.iterdir()
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
        result.append({"name": folder.name, "path": folder, "count": count})
    return result


# ═══════════════════════════════════════════════════════════════════════
#  FINGERPRINT
# ═══════════════════════════════════════════════════════════════════════

def _compute_fp(d: str) -> str:
    parts = []
    for p in sorted(Path(d).rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            s = p.stat()
            parts.append(f"{p}|{s.st_size}|{s.st_mtime:.2f}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()

def _load_fp() -> str:
    try:
        return json.loads(Path(FINGERPRINT_PATH).read_text()).get("fp","")
    except: return ""

def _save_fp(fp: str):
    Path(FINGERPRINT_PATH).write_text(json.dumps({"fp": fp}))

def dataset_changed() -> bool:
    return _compute_fp(TRAINING_DATA_DIR) != _load_fp()


# ═══════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════

def train_model(dataset, lbp, haar, progress_cb=None, force=False) -> bool:
    if os.path.exists(MODEL_PATH) and not force and not dataset_changed():
        if progress_cb: progress_cb("(up-to-date)", 1, 1, 0)
        return True

    faces, labels, label_map = [], [], {}

    for lid, entry in enumerate(dataset):
        name, path, n = entry["name"], entry["path"], 0
        label_map[lid] = name
        for img_p in sorted(path.iterdir()):
            if img_p.suffix.lower() not in IMAGE_EXTS: continue
            img = cv2.imread(str(img_p))
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            roi  = detect_face_forced(gray, lbp, haar)
            if roi is not None:
                faces.append(roi); labels.append(lid); n += 1
        if progress_cb:
            progress_cb(name, lid+1, len(dataset), n)

    if not faces: return False

    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.train(faces, np.array(labels, np.int32))
    rec.save(MODEL_PATH)
    Path(LABELS_PATH).write_text(
        json.dumps({str(k): v for k, v in label_map.items()},
                   ensure_ascii=False, indent=2))
    _save_fp(_compute_fp(TRAINING_DATA_DIR))
    return True


# ═══════════════════════════════════════════════════════════════════════
#  ADD-PERSON WIZARD  (Toplevel)
# ═══════════════════════════════════════════════════════════════════════

class AddPersonWizard(tk.Toplevel):
    """
    Step 1 – Name input
    Step 2a – Camera capture  (live preview, auto-capture N frames)
    Step 2b – Import from folder/files
    """

    def __init__(self, parent: "FaceRecognitionApp"):
        super().__init__(parent)
        self.app = parent
        self.title("➕  Add New Person")
        self.configure(bg=BG_DARK)
        self.resizable(False, False)
        self.grab_set()             # modal

        self.person_name  = tk.StringVar()
        self.capture_cap  = None
        self.capture_run  = False
        self.captured     = 0
        self.save_dir     = None

        self._build_step1()
        self.geometry("520x320")
        self.protocol("WM_DELETE_WINDOW", self._cleanup_close)

    # ── Step 1: name ────────────────────────────────────────────────────

    def _build_step1(self):
        self._clear()
        self.geometry("520x260")

        tk.Label(self, text="New Person — Enter Name",
                 font=FT_TITLE, bg=BG_DARK, fg=ACCENT).pack(pady=(24,4))
        tk.Label(self, text="Use the exact name you want displayed on matches.",
                 font=FT_SMALL, bg=BG_DARK, fg=TEXT_DIM).pack()

        entry_frame = tk.Frame(self, bg=BG_DARK)
        entry_frame.pack(pady=16)
        tk.Label(entry_frame, text="Name:", font=FT, bg=BG_DARK, fg=TEXT,
                 width=7, anchor="e").pack(side="left")
        self.name_entry = tk.Entry(entry_frame, textvariable=self.person_name,
                                   font=FT, bg=BG_INPUT, fg=TEXT,
                                   insertbackground=TEXT, relief="flat",
                                   width=28)
        self.name_entry.pack(side="left", padx=8, ipady=5)
        self.name_entry.focus_set()
        self.name_entry.bind("<Return>", lambda _: self._step1_next())

        btn_row = tk.Frame(self, bg=BG_DARK)
        btn_row.pack()
        self._btn(btn_row, "Cancel", BG_CARD, self._cleanup_close).pack(side="left", padx=6)
        self._btn(btn_row, "Next →", ACCENT,  self._step1_next, dark=True).pack(side="left")

    def _step1_next(self):
        name = self.person_name.get().strip()
        if not name:
            messagebox.showwarning("Name required", "Please enter a name.", parent=self)
            return
        # Sanitise
        invalid = set('/\\:*?"<>|')
        if any(c in invalid for c in name):
            messagebox.showwarning("Invalid name",
                                   "Name may not contain: / \\ : * ? \" < > |",
                                   parent=self)
            return
        self.save_dir = Path(TRAINING_DATA_DIR) / name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._build_step2(name)

    # ── Step 2: choose method ────────────────────────────────────────────

    def _build_step2(self, name: str):
        self._clear()
        self.geometry("520x280")

        tk.Label(self, text=f'Add images for  "{name}"',
                 font=FT_TITLE, bg=BG_DARK, fg=ACCENT).pack(pady=(24, 4))
        tk.Label(self, text=f"Target folder:  training-data/{name}/",
                 font=FT_SMALL, bg=BG_DARK, fg=TEXT_DIM).pack()

        btn_row = tk.Frame(self, bg=BG_DARK)
        btn_row.pack(pady=22)

        self._btn(btn_row, "📸  Capture from Camera",
                  ACCENT2, self._start_capture, dark=True,
                  pad=(18, 10)).pack(side="left", padx=10)
        self._btn(btn_row, "📁  Import from Files",
                  ACCENT3, self._import_files, dark=True,
                  pad=(18, 10)).pack(side="left", padx=10)

        tk.Label(self, text="Capture: auto-takes 40 photos  |  Import: pick any image files",
                 font=FT_SMALL, bg=BG_DARK, fg=TEXT_DIM).pack()

        self._btn(self, "← Back", BG_CARD, self._build_step1).pack(pady=12)

    # ── Camera capture ───────────────────────────────────────────────────

    def _start_capture(self):
        self._clear()
        self.geometry("620x540")

        tk.Label(self, text=f"📸  Capturing: {self.person_name.get().strip()}",
                 font=FT_TITLE, bg=BG_DARK, fg=ACCENT).pack(pady=(10, 4))

        self.cap_canvas = tk.Label(self, bg="#000000")
        self.cap_canvas.pack(padx=8, pady=4)

        info_row = tk.Frame(self, bg=BG_DARK)
        info_row.pack()
        self.cap_count_var = tk.StringVar(value="0 / 40 captured")
        tk.Label(info_row, textvariable=self.cap_count_var,
                 font=FT_BOLD, bg=BG_DARK, fg=ACCENT2).pack(side="left", padx=12)
        self.cap_pbar = ttk.Progressbar(info_row, maximum=CAPTURE_SAMPLES,
                                         length=220, mode="determinate")
        self.cap_pbar.pack(side="left", padx=8)

        self.cap_status = tk.Label(self, text="Opening camera …",
                                   font=FT_SMALL, bg=BG_DARK, fg=TEXT_DIM)
        self.cap_status.pack()

        btn_row = tk.Frame(self, bg=BG_DARK)
        btn_row.pack(pady=8)
        self.cap_btn = self._btn(btn_row, "⏸  Pause", BG_CARD,
                                  self._toggle_capture)
        self.cap_btn.pack(side="left", padx=8)
        self._btn(btn_row, "✔  Done", ACCENT2,
                  self._finish_capture, dark=True).pack(side="left", padx=8)
        self._btn(btn_row, "Cancel", WARN,
                  self._cleanup_close, dark=True).pack(side="left")

        # Open camera (share main app camera if idle, else new)
        self.capture_cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not self.capture_cap.isOpened():
            messagebox.showerror("Camera Error",
                                 "Cannot open webcam.", parent=self)
            return
        self.capture_cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.capture_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.captured     = 0
        self.capture_run  = True
        self._last_capture_time = 0
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def _capture_loop(self):
        lbp = self.app.lbp_cascade
        haar= self.app.haar_cascade
        while self.capture_run and self.captured < CAPTURE_SAMPLES:
            ret, frame = self.capture_cap.read()
            if not ret: break

            display = enhance_frame(frame.copy())
            now_ms  = int(time.time() * 1000)

            # Auto-capture every CAPTURE_INTERVAL_MS
            if now_ms - self._last_capture_time >= CAPTURE_INTERVAL_MS:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi  = detect_face_forced(gray, lbp, haar)
                if roi is not None:
                    idx = self.captured + 1
                    fname = self.save_dir / f"capture_{idx:04d}.jpg"
                    cv2.imwrite(str(fname), roi)
                    self.captured += 1
                    self._last_capture_time = now_ms
                    self.after(0, self._update_capture_ui)

            # Overlay progress on preview
            cv2.putText(display,
                        f"{self.captured}/{CAPTURE_SAMPLES} saved",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (50, 220, 60), 2)

            # Resize for preview canvas (420×315)
            h, w = display.shape[:2]
            pw, ph = 420, int(420 * h / w)
            preview = cv2.resize(display, (pw, ph), interpolation=cv2.INTER_LANCZOS4)
            rgb     = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            photo   = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self.after(0, lambda p=photo: self._update_cap_canvas(p))

        if self.captured >= CAPTURE_SAMPLES:
            self.after(0, lambda: self.cap_status.config(
                text="✔  Capture complete! Click Done.", fg=ACCENT2))

    def _update_cap_canvas(self, photo):
        self.cap_canvas.configure(image=photo)
        self.cap_canvas.image = photo

    def _update_capture_ui(self):
        self.cap_count_var.set(f"{self.captured} / {CAPTURE_SAMPLES} captured")
        self.cap_pbar.config(value=self.captured)

    def _toggle_capture(self):
        self.capture_run = not self.capture_run
        self.cap_btn.config(
            text="▶  Resume" if not self.capture_run else "⏸  Pause")
        if self.capture_run:
            threading.Thread(target=self._capture_loop, daemon=True).start()

    def _finish_capture(self):
        self.capture_run = False
        if self.capture_cap:
            self.capture_cap.release()
        n = self.captured
        if n == 0:
            messagebox.showwarning("No images", "No images were captured.", parent=self)
            return
        self._finalize(n)

    # ── Import from files ────────────────────────────────────────────────

    def _import_files(self):
        files = filedialog.askopenfilenames(
            parent=self,
            title="Select images for " + self.person_name.get().strip(),
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
                       ("All files", "*.*")]
        )
        if not files:
            return

        copied = 0
        for src in files:
            src_p = Path(src)
            dst   = self.save_dir / f"img_{copied+1:04d}{src_p.suffix.lower()}"
            shutil.copy2(str(src_p), str(dst))
            copied += 1

        self._finalize(copied)

    # ── Finalize ─────────────────────────────────────────────────────────

    def _finalize(self, n: int):
        """Called after images are saved. Refreshes parent + triggers retrain."""
        self._cleanup_close()
        # Refresh dataset panel and retrain in parent
        self.app.after(0, lambda: self.app._on_person_added(
            self.person_name.get().strip(), n))

    # ── Helpers ──────────────────────────────────────────────────────────

    def _clear(self):
        for w in self.winfo_children():
            w.destroy()

    def _cleanup_close(self):
        self.capture_run = False
        if self.capture_cap and self.capture_cap.isOpened():
            self.capture_cap.release()
        self.grab_release()
        self.destroy()

    @staticmethod
    def _btn(parent, text, color, cmd, dark=False, pad=(10, 5)):
        return tk.Button(parent, text=text, font=FT,
                         bg=color, fg="#0d1117" if dark else TEXT,
                         relief="flat", cursor="hand2",
                         padx=pad[0], pady=pad[1], command=cmd)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════

class FaceRecognitionApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Face Recognition Pro  v3")
        self.configure(bg=BG_DARK)
        self.resizable(True, True)
        self.minsize(1140, 700)

        self.dataset      = []
        self.label_map    = {}
        self.recognizer   = None
        self.lbp_cascade  = None
        self.haar_cascade = None
        self.cap          = None
        self.browser_opened = False
        self._cam_running   = False

        self._build_ui()
        self.after(200, self._boot)

    # ────────────────────────────────────────────────────────────────────
    #  UI CONSTRUCTION
    # ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Title bar
        bar = tk.Frame(self, bg=BG_MID, height=54)
        bar.pack(fill="x")
        bar.pack_propagate(False)
        tk.Label(bar, text="  🎯  FACE RECOGNITION PRO",
                 font=("Segoe UI", 14, "bold"),
                 bg=BG_MID, fg=ACCENT).pack(side="left", padx=16, pady=14)
        tk.Label(bar, text="v3  •  github.com/thelastx7",
                 font=FT_SMALL, bg=BG_MID, fg=TEXT_DIM).pack(side="right", padx=16)

        # Main row
        main = tk.Frame(self, bg=BG_DARK)
        main.pack(fill="both", expand=True, padx=10, pady=(6, 4))

        left = tk.Frame(main, bg=BG_DARK, width=390)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)
        self._build_left(left)

        right = tk.Frame(main, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True)
        self._build_camera(right)

        # Status bar
        sb = tk.Frame(self, bg=BG_MID, height=32)
        sb.pack(fill="x", side="bottom")
        self.status_var = tk.StringVar(value="Initialising …")
        tk.Label(sb, textvariable=self.status_var, font=FT_SMALL,
                 bg=BG_MID, fg=TEXT_DIM, anchor="w").pack(
                 side="left", padx=12, pady=6)
        self.url_lbl = tk.Label(sb, text="", font=FT_SMALL,
                                bg=BG_MID, fg=ACCENT2)
        self.url_lbl.pack(side="right", padx=12)

    # ── Left panel ───────────────────────────────────────────────────────

    def _build_left(self, parent):

        # ── Dataset card ─────────────────────────────────────────────────
        ds = tk.Frame(parent, bg=BG_CARD)
        ds.pack(fill="x", pady=(0, 8))

        hdr = tk.Frame(ds, bg=BG_CARD)
        hdr.pack(fill="x", padx=12, pady=(10, 0))
        tk.Label(hdr, text="📂  DATASET  —  BOUCLE",
                 font=FT_TITLE, bg=BG_CARD, fg=ACCENT).pack(side="left")
        self._btn_add = tk.Button(
            hdr, text="➕  Add Person",
            font=FT_SMALL, bg=ACCENT2, fg="#0d1117",
            relief="flat", cursor="hand2", padx=8, pady=3,
            command=self._open_add_wizard)
        self._btn_add.pack(side="right")

        self.ds_summary = tk.Label(ds, text="Scanning …",
                                   font=FT_SMALL, bg=BG_CARD, fg=TEXT_DIM, anchor="w")
        self.ds_summary.pack(fill="x", padx=12, pady=(2, 4))
        tk.Frame(ds, bg=BG_MID, height=1).pack(fill="x", padx=8)

        lf = tk.Frame(ds, bg=BG_CARD)
        lf.pack(fill="both", pady=4)
        sb2 = tk.Scrollbar(lf, bg=BG_MID, troughcolor=BG_DARK, width=8)
        sb2.pack(side="right", fill="y")
        self.ds_canvas = tk.Canvas(lf, bg=BG_CARD, height=265,
                                   highlightthickness=0,
                                   yscrollcommand=sb2.set)
        self.ds_canvas.pack(side="left", fill="both", expand=True)
        sb2.config(command=self.ds_canvas.yview)
        self.ds_inner = tk.Frame(self.ds_canvas, bg=BG_CARD)
        self._ds_win  = self.ds_canvas.create_window(
            (0, 0), window=self.ds_inner, anchor="nw")
        self.ds_inner.bind("<Configure>",
                           lambda e: self.ds_canvas.configure(
                               scrollregion=self.ds_canvas.bbox("all")))
        self.ds_canvas.bind("<Configure>",
                            lambda e: self.ds_canvas.itemconfig(
                                self._ds_win, width=e.width))

        # ── Training card ─────────────────────────────────────────────────
        tr = tk.Frame(parent, bg=BG_CARD)
        tr.pack(fill="x", pady=(0, 8))

        tk.Label(tr, text="🧠  TRAINING",
                 font=FT_TITLE, bg=BG_CARD, fg=ACCENT, anchor="w"
                 ).pack(fill="x", padx=12, pady=(10, 2))

        self.train_status = tk.Label(tr, text="Waiting …",
                                     font=FT_SMALL, bg=BG_CARD, fg=TEXT_DIM, anchor="w")
        self.train_status.pack(fill="x", padx=12)

        self.train_bar = ttk.Progressbar(tr, mode="determinate", maximum=100)
        self.train_bar.pack(fill="x", padx=12, pady=6)

        self.train_detail = tk.Label(tr, text="", font=FT_SMALL,
                                     bg=BG_CARD, fg=TEXT_DIM, anchor="w")
        self.train_detail.pack(fill="x", padx=12, pady=(0, 4))

        br = tk.Frame(tr, bg=BG_CARD)
        br.pack(fill="x", padx=12, pady=(0, 10))
        self.btn_train = tk.Button(
            br, text="⟳  Force Retrain",
            font=FT, bg=ACCENT, fg="#0d1117",
            relief="flat", cursor="hand2", padx=10, pady=5,
            command=self._force_retrain)
        self.btn_train.pack(side="left")

        # ── Recognition card ──────────────────────────────────────────────
        rc = tk.Frame(parent, bg=BG_CARD)
        rc.pack(fill="x")

        tk.Label(rc, text="📷  RECOGNITION",
                 font=FT_TITLE, bg=BG_CARD, fg=ACCENT, anchor="w"
                 ).pack(fill="x", padx=12, pady=(10, 2))

        self.rec_status = tk.Label(rc, text="Waiting for model …",
                                   font=FT_SMALL, bg=BG_CARD, fg=TEXT_DIM, anchor="w")
        self.rec_status.pack(fill="x", padx=12)

        # Live confidence threshold slider
        slider_row = tk.Frame(rc, bg=BG_CARD)
        slider_row.pack(fill="x", padx=12, pady=(6, 0))
        tk.Label(slider_row, text="Sensitivity:", font=FT_SMALL,
                 bg=BG_CARD, fg=TEXT_DIM).pack(side="left")
        self.conf_var = tk.DoubleVar(value=CONFIDENCE_THRESHOLD)
        self.conf_slider = ttk.Scale(
            slider_row, from_=40, to=130, variable=self.conf_var,
            orient="horizontal", length=160,
            command=lambda _: self.conf_val_lbl.config(
                text=f"{self.conf_var.get():.0f}"))
        self.conf_slider.pack(side="left", padx=6)
        self.conf_val_lbl = tk.Label(
            slider_row, text=f"{CONFIDENCE_THRESHOLD:.0f}",
            font=FT_MONO, bg=BG_CARD, fg=ACCENT, width=4)
        self.conf_val_lbl.pack(side="left")
        tk.Label(slider_row, text="← strict   loose →",
                 font=("Segoe UI", 8), bg=BG_CARD, fg=TEXT_DIM).pack(side="left", padx=4)

        self.match_lbl = tk.Label(rc, text="",
                                  font=("Segoe UI", 11, "bold"),
                                  bg=BG_CARD, fg=ACCENT2, anchor="w")
        self.match_lbl.pack(fill="x", padx=12, pady=(4, 2))

        # Shows closest guess even when NO MATCH
        self.closest_lbl = tk.Label(rc, text="",
                                    font=FT_SMALL, bg=BG_CARD, fg=TEXT_DIM, anchor="w")
        self.closest_lbl.pack(fill="x", padx=12, pady=(0, 10))

    # ── Camera panel ─────────────────────────────────────────────────────

    def _build_camera(self, parent):
        cam = tk.Frame(parent, bg=BG_CARD)
        cam.pack(fill="both", expand=True)
        tk.Label(cam, text="📡  LIVE FEED",
                 font=FT_TITLE, bg=BG_CARD, fg=ACCENT, anchor="w"
                 ).pack(fill="x", padx=12, pady=(10, 4))
        self.cam_lbl = tk.Label(cam, bg="#000000")
        self.cam_lbl.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        ph = np.zeros((480, 640, 3), np.uint8)
        cv2.putText(ph, "Camera initialising ...", (60, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (88, 166, 255), 2)
        self._show_frame(ph)

    # ────────────────────────────────────────────────────────────────────
    #  BOOT
    # ────────────────────────────────────────────────────────────────────

    def _boot(self):
        self._set_status("Loading cascades …")
        self.lbp_cascade, self.haar_cascade = load_cascades()
        self._set_status("Scanning dataset …")
        self.dataset = scan_dataset(TRAINING_DATA_DIR)
        self._populate_boucle()
        if not self.dataset:
            messagebox.showwarning("Empty dataset",
                                   f"No person folders in '{TRAINING_DATA_DIR}'.\n"
                                   "Use  ➕ Add Person  to create one.")
        self._set_status("Training model …")
        threading.Thread(target=self._run_training, daemon=True).start()

    # ────────────────────────────────────────────────────────────────────
    #  DATASET BOUCLE
    # ────────────────────────────────────────────────────────────────────

    def _populate_boucle(self):
        for w in self.ds_inner.winfo_children():
            w.destroy()
        if not self.dataset:
            tk.Label(self.ds_inner,
                     text="  No persons yet — click ➕ Add Person",
                     font=FT_SMALL, bg=BG_CARD, fg=TEXT_DIM).pack(anchor="w", padx=8)
            self.ds_summary.config(text="0 persons  •  0 images")
            return

        total = sum(d["count"] for d in self.dataset)
        n     = len(self.dataset)
        mx    = max(d["count"] for d in self.dataset)
        self.ds_summary.config(
            text=f"{n} persons  •  {total} images  •  avg {total/max(n,1):.1f}/person")

        for i, entry in enumerate(self.dataset):
            self.after(i * 40,
                       lambda e=entry, idx=i, m=mx: self._add_row(e, idx, m))

    def _add_row(self, entry, idx, max_count):
        name, count = entry["name"], entry["count"]
        BAR_W = 110
        fill  = int(count / max(max_count, 1) * BAR_W)

        row = tk.Frame(self.ds_inner, bg=BG_CARD)
        row.pack(fill="x", padx=6, pady=1)

        tk.Label(row, text=f"{idx+1:02d}",
                 font=FT_MONO, bg=BG_CARD, fg=TEXT_DIM,
                 width=3, anchor="e").pack(side="left", padx=(4, 6))

        # ⚠️  warn when person has fewer than 20 training images
        warn_icon = "⚠️ " if count < 20 else "   "
        name_color = ACCENT3 if count < 20 else TEXT
        tk.Label(row, text=warn_icon + name, font=FT_SMALL, bg=BG_CARD, fg=name_color,
                 anchor="w", width=22).pack(side="left")
        bc = tk.Canvas(row, width=BAR_W, height=12,
                       bg=BAR_EMPTY, highlightthickness=0)
        bc.pack(side="left", padx=6)
        if fill > 0:
            bc.create_rectangle(0, 0, fill, 12, fill=BAR_FILL, outline="")
        tk.Label(row, text=f"{count}",
                 font=FT_MONO, bg=BG_CARD, fg=ACCENT,
                 width=4, anchor="e").pack(side="left")

        # Right-click → delete
        del_btn = tk.Label(row, text="✕", font=FT_SMALL,
                           bg=BG_CARD, fg=WARN, cursor="hand2")
        del_btn.pack(side="right", padx=4)
        del_btn.bind("<Button-1>",
                     lambda _, n=name, p=entry["path"]: self._delete_person(n, p))

    # ────────────────────────────────────────────────────────────────────
    #  ADD PERSON
    # ────────────────────────────────────────────────────────────────────

    def _open_add_wizard(self):
        # Pause camera while wizard is open to free the device
        self._cam_running = False
        time.sleep(0.15)
        wizard = AddPersonWizard(self)
        self.wait_window(wizard)

    def _on_person_added(self, name: str, n_images: int):
        """Called by wizard after images are saved."""
        self._set_status(f"Added {n_images} images for '{name}' — retraining …")
        self.dataset = scan_dataset(TRAINING_DATA_DIR)
        self._populate_boucle()
        threading.Thread(target=self._run_training,
                         kwargs={"force": True}, daemon=True).start()

    # ────────────────────────────────────────────────────────────────────
    #  DELETE PERSON
    # ────────────────────────────────────────────────────────────────────

    def _delete_person(self, name: str, path: Path):
        if not messagebox.askyesno(
                "Delete person",
                f"Delete all images for  '{name}'?\nThis cannot be undone.",
                parent=self):
            return
        shutil.rmtree(str(path), ignore_errors=True)
        self.dataset = scan_dataset(TRAINING_DATA_DIR)
        self._populate_boucle()
        threading.Thread(target=self._run_training,
                         kwargs={"force": True}, daemon=True).start()

    # ────────────────────────────────────────────────────────────────────
    #  TRAINING
    # ────────────────────────────────────────────────────────────────────

    def _run_training(self, force=False):
        self.after(0, lambda: self.btn_train.config(state="disabled"))
        self.after(0, lambda: self.train_status.config(
            text="Training in progress …", fg=ACCENT))

        def cb(name, done, total, faces):
            pct = int(done / max(total, 1) * 100)
            self.after(0, lambda: self.train_bar.config(value=pct))
            self.after(0, lambda: self.train_detail.config(
                text=f"[{done}/{total}]  {name}  —  {faces} faces"))

        ok = train_model(self.dataset, self.lbp_cascade, self.haar_cascade,
                         progress_cb=cb, force=force)

        if ok:
            self.after(0, lambda: self.train_status.config(
                text="✔  Model ready", fg=ACCENT2))
            self.after(0, lambda: self.train_bar.config(value=100))
            self.after(0, self._load_model_then_camera)
        else:
            self.after(0, lambda: self.train_status.config(
                text="✘  Training failed", fg=WARN))

        self.after(0, lambda: self.btn_train.config(state="normal"))

    def _force_retrain(self):
        self._cam_running = False
        time.sleep(0.15)
        self.dataset = scan_dataset(TRAINING_DATA_DIR)
        self._populate_boucle()
        threading.Thread(target=self._run_training,
                         kwargs={"force": True}, daemon=True).start()

    # ────────────────────────────────────────────────────────────────────
    #  MODEL LOAD
    # ────────────────────────────────────────────────────────────────────

    def _load_model_then_camera(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
            self.rec_status.config(text="Model files missing.", fg=WARN)
            return
        rec = cv2.face.LBPHFaceRecognizer_create()
        try:
            rec.read(MODEL_PATH)
        except cv2.error as e:
            self.rec_status.config(text=f"Load error: {e}", fg=WARN)
            return
        with open(LABELS_PATH, encoding="utf-8") as f:
            self.label_map = {int(k): v for k, v in json.load(f).items()}
        self.recognizer = rec
        self.rec_status.config(
            text=f"✔  {len(self.label_map)} persons loaded", fg=ACCENT2)
        self._set_status("Recognition running — camera live.")
        self._start_camera()

    # ────────────────────────────────────────────────────────────────────
    #  CAMERA
    # ────────────────────────────────────────────────────────────────────

    def _start_camera(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not self.cap.isOpened():
            self.rec_status.config(text="Cannot open webcam.", fg=WARN)
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self._cam_running   = True
        self.browser_opened = False
        threading.Thread(target=self._camera_loop, daemon=True).start()

    def _camera_loop(self):
        """
        Background thread.
        ✔ No flip  → camera is displayed normally (not mirrored)
        ✔ CLAHE + sharpening for quality
        ✔ LBPH recognition on CLAHE-preprocessed ROI
        """
        while self._cam_running:
            if not self.cap or not self.cap.isOpened():
                break
            ret, frame = self.cap.read()
            if not ret: break

            # Quality enhancement — NO cv2.flip call → normal orientation
            display = enhance_frame(frame)

            if self.recognizer and self.lbp_cascade:
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                egray = preprocess_gray(gray)
                faces = self.lbp_cascade.detectMultiScale(
                    egray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

                for (x, y, w, h) in faces:
                    roi  = cv2.resize(egray[y:y+h, x:x+w], (100, 100))
                    # Histogram equalisation on the ROI improves LBPH matching
                    roi  = cv2.equalizeHist(roi)
                    lid, cf = self.recognizer.predict(roi)
                    person  = self.label_map.get(lid, "Unknown")

                    # Read threshold live from slider
                    threshold = self.conf_var.get()

                    if cf < threshold:
                        color = (50, 220, 60)
                        text  = f"MATCH: {person}"
                        (tw, th), _ = cv2.getTextSize(
                            text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                        cv2.rectangle(display,
                                      (x, y-th-16), (x+tw+8, y), color, -1)
                        cv2.putText(display, text, (x+4, y-6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                    (0, 0, 0), 2)
                        cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(display, f"dist {cf:.1f}",
                                    (x+4, y+h+18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        self.after(0, lambda p=person, c=cf:
                                   self.match_lbl.config(
                                       text=f"✔  {p}  (dist {c:.1f})",
                                       fg=ACCENT2))
                        self.after(0, lambda: self.closest_lbl.config(text=""))
                        if not self.browser_opened:
                            self.browser_opened = True
                            webbrowser.open(MATCH_URL)
                            self.after(0, lambda: self.url_lbl.config(
                                text=f"🌐 {MATCH_URL}"))
                    else:
                        color = (40, 40, 220)
                        cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
                        # Show NO MATCH + confidence so user can diagnose
                        cv2.putText(display, f"NO MATCH  dist {cf:.0f}",
                                    (x, y - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                        # Side-panel: show closest guess
                        self.after(0, lambda p=person, c=cf, t=threshold:
                                   self.closest_lbl.config(
                                       text=f"Closest: {p}  dist={c:.0f}  "
                                            f"(need < {t:.0f})",
                                       fg=ACCENT3))
                        self.after(0, lambda: self.match_lbl.config(
                            text="", fg=ACCENT2))

            # HUD
            hud  = "URL opened ✔" if self.browser_opened else "Monitoring ..."
            hcol = (50, 220, 60) if self.browser_opened else (220, 200, 50)
            cv2.putText(display, hud, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, hcol, 2)

            self.after(0, lambda f=display: self._show_frame(f))

        if self.cap:
            self.cap.release()

    # ────────────────────────────────────────────────────────────────────
    #  FRAME DISPLAY
    # ────────────────────────────────────────────────────────────────────

    def _show_frame(self, frame: np.ndarray):
        lw = self.cam_lbl.winfo_width()
        lh = self.cam_lbl.winfo_height()
        if lw < 10 or lh < 10: lw, lh = 640, 420
        fh, fw = frame.shape[:2]
        scale  = min(lw / fw, lh / fh)
        nw, nh = int(fw * scale), int(fh * scale)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
        photo   = ImageTk.PhotoImage(
            image=Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
        self.cam_lbl.configure(image=photo)
        self.cam_lbl.image = photo

    def _set_status(self, msg: str):
        self.status_var.set(f"  {msg}")

    def on_close(self):
        self._cam_running = False
        time.sleep(0.14)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.destroy()


# ═══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def main():
    try:
        from PIL import Image, ImageTk  # noqa
    except ImportError:
        print("[ERROR] Pillow required:  pip install Pillow")
        sys.exit(1)

    Path(TRAINING_DATA_DIR).mkdir(exist_ok=True)

    app = FaceRecognitionApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)

    style = ttk.Style(app)
    style.theme_use("default")
    style.configure("Horizontal.TProgressbar",
                    troughcolor=BAR_EMPTY, background=BAR_FILL, thickness=8)
    app.mainloop()


if __name__ == "__main__":
    main()
