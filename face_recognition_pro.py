"""
face_recognition_pro.py
=======================
Face Recognition System with full Dataset Inspector (Boucle / Loop Display)

Pipeline:
  1. DATASET BOUCLE  — animated loop through every person folder,
                        showing name + image count + progress bar
  2. SMART TRAINING  — retrains ONLY when new images are detected
                        (SHA-256 fingerprint of training directory)
  3. RECOGNITION     — live webcam recognition with LBPH
  4. BROWSER TRIGGER — opens https://github.com/thelastx7 exactly once
                        on the first confirmed face match (boolean guard)
"""

import cv2
import os
import sys
import time
import hashlib
import json
import logging
import webbrowser
import numpy as np
from pathlib import Path
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

TRAINING_DATA_DIR   = "training-data"
CASCADE_PATH        = "opencv-files/lbpcascade_frontalface.xml"
MODEL_PATH          = "modele_visages.yml"
LABELS_PATH         = "modele_visages_labels.json"
FINGERPRINT_PATH    = "training_fingerprint.json"

CONFIDENCE_THRESHOLD = 60.0
DETECTION_SCALE      = 1.2
DETECTION_NEIGHBORS  = 5
WEBCAM_INDEX         = 0

MATCH_URL            = "https://github.com/thelastx7"
IMAGE_EXTS           = {".jpg", ".jpeg", ".png", ".bmp"}

# ═══════════════════════════════════════════════════════════════════════════
#  LOGGING  (clean, timestamped)
# ═══════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  TERMINAL COLOURS  (ANSI – works on Linux / macOS / Windows 10+)
# ═══════════════════════════════════════════════════════════════════════════

class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BLUE   = "\033[94m"
    MAGENTA= "\033[95m"
    WHITE  = "\033[97m"
    DIM    = "\033[2m"


def _bar(filled: int, total: int, width: int = 28) -> str:
    """Return a coloured ASCII progress bar string."""
    ratio   = filled / max(total, 1)
    n_fill  = int(ratio * width)
    n_empty = width - n_fill
    bar     = C.GREEN + "█" * n_fill + C.DIM + "░" * n_empty + C.RESET
    pct     = f"{ratio * 100:5.1f}%"
    return f"[{bar}] {C.YELLOW}{pct}{C.RESET}"


# ═══════════════════════════════════════════════════════════════════════════
#  DATASET INSPECTOR  —  the "BOUCLE" (loop) display
# ═══════════════════════════════════════════════════════════════════════════

def scan_dataset(training_dir: str) -> list[dict]:
    """
    Walk training_dir and return a sorted list of dicts:
        { "name": str, "path": Path, "count": int }
    one entry per person sub-directory.
    """
    root = Path(training_dir)
    if not root.is_dir():
        return []

    results = []
    for d in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not d.is_dir():
            continue
        count = sum(
            1 for f in d.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        )
        results.append({"name": d.name, "path": d, "count": count})
    return results


def display_dataset_boucle(training_dir: str) -> list[dict]:
    """
    ┌──────────────────────────────────────────────────────────┐
    │  Animated loop (boucle) through every person folder.     │
    │  Prints name, image count, and a live progress bar.      │
    │  Returns the dataset info list for use by the trainer.   │
    └──────────────────────────────────────────────────────────┘
    """
    dataset = scan_dataset(training_dir)

    if not dataset:
        print(f"{C.RED}[ERROR] No person folders found in '{training_dir}'.{C.RESET}")
        return []

    total_folders = len(dataset)
    total_images  = sum(d["count"] for d in dataset)
    max_count     = max(d["count"] for d in dataset) if dataset else 1
    max_name_len  = max(len(d["name"]) for d in dataset)

    # ── Header ─────────────────────────────────────────────────────────────
    print()
    print(C.BOLD + C.CYAN + "═" * 66 + C.RESET)
    print(C.BOLD + C.WHITE +
          "   📂  DATASET INSPECTOR  —  BOUCLE / LOOP DISPLAY" + C.RESET)
    print(C.BOLD + C.CYAN + "═" * 66 + C.RESET)
    print(f"   {C.YELLOW}Directory :{C.RESET}  {training_dir}")
    print(f"   {C.YELLOW}Folders   :{C.RESET}  {C.BOLD}{total_folders}{C.RESET} person(s)")
    print(f"   {C.YELLOW}Images    :{C.RESET}  {C.BOLD}{total_images}{C.RESET} total")
    print(C.DIM + "─" * 66 + C.RESET)
    print()

    # ── Boucle (animated loop through every folder) ─────────────────────
    for idx, entry in enumerate(dataset, start=1):
        name  = entry["name"]
        count = entry["count"]
        bar   = _bar(count, max_count)
        rank  = f"{C.DIM}[{idx:02d}/{total_folders:02d}]{C.RESET}"
        label = f"{C.BOLD}{C.WHITE}{name:<{max_name_len}}{C.RESET}"
        imgs  = f"{C.CYAN}{count:>4}{C.RESET} imgs"

        print(f"   {rank}  {label}  {imgs}  {bar}")
        time.sleep(0.03)            # slight delay makes it feel like a live scan

    # ── Summary footer ─────────────────────────────────────────────────────
    print()
    print(C.DIM + "─" * 66 + C.RESET)
    avg = total_images / total_folders if total_folders else 0
    print(f"   {C.GREEN}✔  Scan complete.{C.RESET}  "
          f"Average images per person: {C.BOLD}{avg:.1f}{C.RESET}")
    print(C.BOLD + C.CYAN + "═" * 66 + C.RESET)
    print()

    return dataset


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING-DATA FINGERPRINT  (smart change detection)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_fingerprint(training_dir: str) -> str:
    """SHA-256 over sorted (path, size, mtime) tuples for all images."""
    entries: list[str] = []
    root = Path(training_dir)
    if not root.is_dir():
        return ""
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            s = p.stat()
            entries.append(f"{p.relative_to(root)}|{s.st_size}|{s.st_mtime:.2f}")
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()


def _load_saved_fingerprint() -> str:
    try:
        with open(FINGERPRINT_PATH) as f:
            return json.load(f).get("fingerprint", "")
    except (FileNotFoundError, json.JSONDecodeError):
        return ""


def _save_fingerprint(fp: str) -> None:
    with open(FINGERPRINT_PATH, "w") as f:
        json.dump({"fingerprint": fp}, f)


def training_data_changed() -> bool:
    return _compute_fingerprint(TRAINING_DATA_DIR) != _load_saved_fingerprint()


# ═══════════════════════════════════════════════════════════════════════════
#  FACE DETECTION  (shared by training + recognition)
# ═══════════════════════════════════════════════════════════════════════════

def _load_cascade() -> Optional[cv2.CascadeClassifier]:
    if not os.path.exists(CASCADE_PATH):
        print(f"{C.RED}[ERROR] Cascade not found: {CASCADE_PATH}{C.RESET}")
        return None
    cc = cv2.CascadeClassifier(CASCADE_PATH)
    if cc.empty():
        print(f"{C.RED}[ERROR] Cascade failed to load: {CASCADE_PATH}{C.RESET}")
        return None
    return cc


def _detect_face_roi(
    gray: np.ndarray,
    cascade: cv2.CascadeClassifier,
) -> Optional[np.ndarray]:
    """Return the largest face ROI from a grayscale image, or None."""
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return gray[y : y + h, x : x + w]


# ═══════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_model(dataset: list[dict], force: bool = False) -> bool:
    """
    Train the LBPH recognizer using pre-scanned dataset info.
    Skips if model exists and data hasn't changed (unless force=True).
    """
    model_exists = os.path.exists(MODEL_PATH)

    if model_exists and not force and not training_data_changed():
        print(f"   {C.GREEN}✔  Model is up-to-date — skipping retraining.{C.RESET}\n")
        return True

    cascade = _load_cascade()
    if cascade is None:
        return False

    print(C.BOLD + C.CYAN + "═" * 66 + C.RESET)
    print(C.BOLD + C.WHITE + "   🧠  TRAINING  —  BOUCLE THROUGH DATASET" + C.RESET)
    print(C.BOLD + C.CYAN + "═" * 66 + C.RESET)

    faces:     list[np.ndarray] = []
    labels:    list[int]        = []
    label_map: dict[int, str]   = {}

    for label_id, entry in enumerate(dataset):
        name  = entry["name"]
        path  = entry["path"]
        loaded = 0
        skipped = 0

        label_map[label_id] = name

        for img_path in sorted(path.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            roi  = _detect_face_roi(gray, cascade)
            if roi is not None:
                faces.append(roi)
                labels.append(label_id)
                loaded += 1
            else:
                skipped += 1

        status = C.GREEN + "✔" + C.RESET if loaded > 0 else C.RED + "✘" + C.RESET
        skip_note = f"  {C.DIM}({skipped} skipped){C.RESET}" if skipped else ""
        print(f"   {status}  {C.BOLD}{name:<30}{C.RESET}"
              f"  {C.CYAN}{loaded:>3} faces{C.RESET}{skip_note}")

    print()

    if not faces:
        print(f"{C.RED}[ERROR] No usable faces found — training aborted.{C.RESET}")
        return False

    print(f"   {C.YELLOW}Training LBPH on {len(faces)} samples …{C.RESET}", end="", flush=True)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels, dtype=np.int32))
    recognizer.save(MODEL_PATH)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in label_map.items()}, f,
                  ensure_ascii=False, indent=2)
    _save_fingerprint(_compute_fingerprint(TRAINING_DATA_DIR))

    print(f"  {C.GREEN}Done!{C.RESET}")
    print(f"   Model  → {C.BOLD}{MODEL_PATH}{C.RESET}")
    print(f"   Labels → {C.BOLD}{LABELS_PATH}{C.RESET}")
    print(C.BOLD + C.CYAN + "═" * 66 + C.RESET)
    print()
    return True


# ═══════════════════════════════════════════════════════════════════════════
#  RECOGNITION  (webcam loop)
# ═══════════════════════════════════════════════════════════════════════════

def _load_recognizer_and_labels() -> tuple[
    Optional[cv2.face.LBPHFaceRecognizer], dict[int, str]
]:
    for path, label in [(MODEL_PATH, "Model"), (LABELS_PATH, "Labels")]:
        if not os.path.exists(path):
            print(f"{C.RED}[ERROR] {label} file missing: {path}{C.RESET}")
            return None, {}

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        recognizer.read(MODEL_PATH)
    except cv2.error as exc:
        print(f"{C.RED}[ERROR] Cannot read model: {exc}{C.RESET}")
        return None, {}

    with open(LABELS_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    label_map = {int(k): v for k, v in raw.items()}
    return recognizer, label_map


def run_recognition() -> None:
    """
    Live webcam recognition loop.

    – Green box  → confirmed match (confidence < threshold)
    – Red box    → unknown face
    – Browser opens https://github.com/thelastx7 on FIRST match only
      (browser_opened boolean flag prevents multiple tabs)
    """
    recognizer, label_map = _load_recognizer_and_labels()
    if recognizer is None:
        return

    cascade = _load_cascade()
    if cascade is None:
        return

    print(C.BOLD + C.CYAN + "═" * 66 + C.RESET)
    print(C.BOLD + C.WHITE + "   📷  RECOGNITION  —  LIVE WEBCAM" + C.RESET)
    print(C.BOLD + C.CYAN + "═" * 66 + C.RESET)
    print(f"   Known persons  : {C.BOLD}{len(label_map)}{C.RESET}")
    print(f"   Confidence cap : {C.BOLD}{CONFIDENCE_THRESHOLD}{C.RESET} "
          f"{C.DIM}(lower = stricter){C.RESET}")
    print(f"   On match URL   : {C.CYAN}{MATCH_URL}{C.RESET}")
    print(C.DIM + "   Press  Q  to quit." + C.RESET)
    print(C.BOLD + C.CYAN + "═" * 66 + C.RESET)
    print()

    # ── Open webcam ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(
            f"{C.RED}[ERROR] Cannot open webcam (index {WEBCAM_INDEX}).\n"
            f"        Check the camera is connected and not used by another app.{C.RESET}"
        )
        return

    # ── One-shot flag: prevents opening more than one browser tab ──────────
    browser_opened: bool = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("Webcam frame grab failed — stopping.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=DETECTION_SCALE,
                minNeighbors=DETECTION_NEIGHBORS,
                minSize=(60, 60),
            )

            for (x, y, w, h) in faces:
                roi             = gray[y : y + h, x : x + w]
                label_id, conf  = recognizer.predict(roi)
                person          = label_map.get(label_id, "Unknown")
                is_match        = conf < CONFIDENCE_THRESHOLD

                if is_match:
                    color = (0, 220, 60)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"MATCH: {person}",
                                (x, y - 22), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, color, 2)
                    cv2.putText(frame, f"conf {conf:.1f}",
                                (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 1)

                    # ── Boolean guard: open URL once per session ───────────
                    if not browser_opened:
                        log.info("✔ Match: %s (conf %.1f) → opening %s",
                                 person, conf, MATCH_URL)
                        webbrowser.open(MATCH_URL)
                        browser_opened = True        # ← flipped; never opens again

                else:
                    color = (0, 0, 220)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, "NO MATCH",
                                (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, color, 2)

            # ── HUD overlay ───────────────────────────────────────────────
            hud = "URL sent ✔" if browser_opened else "Monitoring..."
            hud_color = (60, 220, 60) if browser_opened else (220, 220, 60)
            cv2.putText(frame, hud, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, hud_color, 2)

            cv2.imshow("Face Recognition Pro  —  Q to quit", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                log.info("User quit.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n   {C.GREEN}Camera released. Session ended.{C.RESET}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    os.system("cls" if os.name == "nt" else "clear")

    print()
    print(C.BOLD + C.MAGENTA +
          "  ███████╗ █████╗  ██████╗███████╗    ██████╗ ███████╗ ██████╗" + C.RESET)
    print(C.BOLD + C.MAGENTA +
          "  ██╔════╝██╔══██╗██╔════╝██╔════╝    ██╔══██╗██╔════╝██╔════╝" + C.RESET)
    print(C.BOLD + C.MAGENTA +
          "  █████╗  ███████║██║     █████╗      ██████╔╝█████╗  ██║     " + C.RESET)
    print(C.BOLD + C.MAGENTA +
          "  ██╔══╝  ██╔══██║██║     ██╔══╝      ██╔══██╗██╔══╝  ██║     " + C.RESET)
    print(C.BOLD + C.MAGENTA +
          "  ██║     ██║  ██║╚██████╗███████╗    ██║  ██║███████╗╚██████╗" + C.RESET)
    print(C.BOLD + C.MAGENTA +
          "  ╚═╝     ╚═╝  ╚═╝ ╚═════╝╚══════╝    ╚═╝  ╚═╝╚══════╝ ╚═════╝" + C.RESET)
    print(C.DIM + "                       PRO  •  opencv-lbph  •  github.com/thelastx7" + C.RESET)
    print()
    time.sleep(0.6)

    # ── Step 1: Dataset boucle (animated scan) ─────────────────────────────
    dataset = display_dataset_boucle(TRAINING_DATA_DIR)
    if not dataset:
        sys.exit(1)

    # ── Step 2: Smart training ─────────────────────────────────────────────
    ok = train_model(dataset, force=False)
    if not ok:
        print(f"{C.RED}Training failed — exiting.{C.RESET}")
        sys.exit(1)

    # ── Step 3: Live recognition ───────────────────────────────────────────
    run_recognition()


if __name__ == "__main__":
    main()
