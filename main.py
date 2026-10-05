import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import os
from pathlib import Path
import json
import tkinter as tk
from tkinter import messagebox
import threading
import queue
import math
import io
import wave
import msvcrt
import tempfile
from datetime import datetime
import winsound

# ============================================================
# FACE RECOGNITION SYSTEM
# Face detection / dataset / LBPH recognition logic is retained.
# ============================================================

DATASET_PATH = "datasets"
MODEL_PATH = "Trainer.yml"
ID_TO_NAME_FILE = "id_to_name.json"
MIN_IMAGES_PER_USER = 500
CONFIDENCE_THRESHOLD = 60


def acquire_single_instance_lock():
    lock_path = os.path.join(tempfile.gettempdir(), "face_recognition_system.lock")
    lock_file = open(lock_path, "a+b")
    lock_file.seek(0, os.SEEK_END)
    if lock_file.tell() == 0:
        lock_file.write(b"\0")
        lock_file.flush()
    lock_file.seek(0)
    try:
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        lock_file.close()
        return None
    return lock_file


def play_alert_sound():
    try:
        sample_rate = 22050
        duration = 2.1
        times = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
        sweep_position = np.mod(times, 0.7) / 0.7
        sweep = np.where(sweep_position < 0.5, sweep_position * 2, (1 - sweep_position) * 2)
        frequency = 650 + sweep * 900
        phase = 2 * np.pi * np.cumsum(frequency) / sample_rate
        pulse = np.where(np.mod(times, 0.35) < 0.23, 1.0, 0.35)
        envelope = np.minimum(1.0, np.minimum(times / 0.04, (duration - times) / 0.08))
        samples = (np.sin(phase) * pulse * envelope * 0.85 * 32767).astype(np.int16)

        audio = io.BytesIO()
        with wave.open(audio, "wb") as sound_file:
            sound_file.setnchannels(1)
            sound_file.setsampwidth(2)
            sound_file.setframerate(sample_rate)
            sound_file.writeframes(samples.tobytes())
        winsound.PlaySound(audio.getvalue(), winsound.SND_MEMORY)
    except Exception:
        try:
            winsound.Beep(1000, 250)
            winsound.Beep(750, 250)
            winsound.Beep(1000, 400)
        except Exception:
            pass


def get_face_cascade():
    candidates = []
    base = getattr(cv2, "data", None)
    if base is not None:
        candidates.append(os.path.join(base.haarcascades, "haarcascade_frontalface_default.xml"))
        candidates.append(
            os.path.join(str(Path(base.__file__).resolve().parent), "data", "haarcascade_frontalface_default.xml")
        )

    candidates.extend([
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Python", "Python313", "Lib", "site-packages", "cv2", "data", "haarcascade_frontalface_default.xml"),
        os.path.join(os.environ.get("USERPROFILE", ""), "AppData", "Roaming", "Python", "Python313", "site-packages", "cv2", "data", "haarcascade_frontalface_default.xml"),
        os.path.join(os.environ.get("PROGRAMFILES", ""), "OpenCV", "etc", "haarcascades", "haarcascade_frontalface_default.xml"),
    ])

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            cascade = cv2.CascadeClassifier(candidate)
            if not cascade.empty():
                return cascade

    fallback = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(fallback)
    if not cascade.empty():
        return cascade
    raise FileNotFoundError("Unable to find Haar cascade file for face detection.")


def augment_images(image):
    augmented_images = [image]
    augmented_images.append(cv2.flip(image, 1))
    for angle in [-15, 15]:
        rows, cols = image.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated = cv2.warpAffine(image, M, (cols, rows))
        augmented_images.append(rotated)
    return augmented_images


def train_recognizer(progress_callback=None):
    """Train the original LBPH recognizer. Optional callback is UI-only."""
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids, user_to_id = [], [], {}
    user_id_counter = 1

    if progress_callback:
        progress_callback(10, "Loading face datasets...")

    if not os.path.exists(DATASET_PATH):
        print("Dataset not found.")
        if progress_callback:
            progress_callback(0, "Dataset not found")
        return False

    folders = []
    for dirpath, _, filenames in os.walk(DATASET_PATH):
        user_folder = os.path.basename(dirpath)
        user_faces = []
        for filename in filenames:
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                try:
                    path = os.path.join(dirpath, filename)
                    img = Image.open(path).convert("L")
                    user_faces.append(np.array(img))
                except Exception as e:
                    print(f"Error reading image: {e}")
        if user_faces:
            folders.append((user_folder, user_faces))

    if progress_callback:
        progress_callback(30, "Preparing training data...")

    for index, (user_folder, user_faces) in enumerate(folders):
        if user_folder not in user_to_id:
            user_to_id[user_folder] = user_id_counter
            user_id_counter += 1

        user_id = user_to_id[user_folder]
        while len(user_faces) < MIN_IMAGES_PER_USER:
            original_count = len(user_faces)
            for face in user_faces[:original_count]:
                user_faces.extend(augment_images(face))
                if len(user_faces) >= MIN_IMAGES_PER_USER:
                    break
            if original_count == len(user_faces):
                break

        faces.extend(user_faces[:MIN_IMAGES_PER_USER])
        ids.extend([user_id] * MIN_IMAGES_PER_USER)
        pct = 30 + int(((index + 1) / max(len(folders), 1)) * 25)
        if progress_callback:
            progress_callback(pct, "Preparing training data...")

    if not faces or not ids:
        print("No faces found for training.")
        if progress_callback:
            progress_callback(0, "No faces found for training")
        return False

    if progress_callback:
        progress_callback(65, "Training model...")
    recognizer.train(faces, np.array(ids))

    if progress_callback:
        progress_callback(88, "Saving trained model...")
    recognizer.write(MODEL_PATH)
    with open(ID_TO_NAME_FILE, "w") as f:
        json.dump(user_to_id, f)

    if progress_callback:
        progress_callback(100, "Training completed successfully!")
    print("Training completed.")
    return True



class RoundedButton(tk.Canvas):
    """Responsive rounded button with centered text and hover effect."""

    def __init__(self, master, text="", command=None, width=190, height=44,
                 radius=16, bg="#0d2940", hover_bg="#123d5d",
                 fg="#f2f7fb", font=("Segoe UI", 10, "bold"),
                 border="#19c8ff", border_width=2, **kwargs):
        super().__init__(
            master, width=width, height=height,
            bg=master.cget("bg"), highlightthickness=0, bd=0,
            relief="flat", **kwargs
        )
        self.button_text = text
        self.command = command
        self.normal_bg = bg
        self.hover_bg = hover_bg
        self.fg = fg
        self.font = font
        self.border = border
        self.border_width = border_width
        self.radius = radius
        self._current_bg = bg

        self.bind("<Configure>", self._redraw)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
        self._redraw()

    def _points(self, x1, y1, x2, y2, r):
        return [
            x1+r,y1, x2-r,y1, x2,y1, x2,y1+r,
            x2,y2-r, x2,y2, x2-r,y2, x1+r,y2,
            x1,y2, x1,y2-r, x1,y1+r, x1,y1
        ]

    def _redraw(self, event=None):
        self.delete("all")
        w = max(2, self.winfo_width())
        h = max(2, self.winfo_height())
        r = min(self.radius, (h-2)//2, (w-2)//2)

        self.create_polygon(
            self._points(1, 1, w-1, h-1, r),
            smooth=True, fill=self.border, outline=self.border
        )

        p = self.border_width
        ir = max(3, r-p)
        self.create_polygon(
            self._points(p+1, p+1, w-p-1, h-p-1, ir),
            smooth=True, fill=self._current_bg, outline=self._current_bg
        )

        self.create_text(
            w / 2, h / 2,
            text=self.button_text,
            fill=self.fg,
            font=self.font,
            anchor="center"
        )

    def _on_enter(self, event=None):
        self._current_bg = self.hover_bg
        self._redraw()

    def _on_leave(self, event=None):
        self._current_bg = self.normal_bg
        self._redraw()

    def _on_click(self, event=None):
        if callable(self.command):
            self.command()

    def configure(self, cnf=None, **kwargs):
        if cnf:
            kwargs.update(cnf)
        if "text" in kwargs:
            self.button_text = kwargs.pop("text")
        if "bg" in kwargs:
            self.normal_bg = kwargs.pop("bg")
            self._current_bg = self.normal_bg
        if "hover_bg" in kwargs:
            self.hover_bg = kwargs.pop("hover_bg")
        if "fg" in kwargs:
            self.fg = kwargs.pop("fg")
        if "font" in kwargs:
            self.font = kwargs.pop("font")
        result = super().configure(**kwargs)
        self._redraw()
        return result

    config = configure


class FaceRecognitionApp:
    # Reference-image palette
    BG = "#020b14"
    HEADER = "#071523"
    PANEL = "#071827"
    PANEL_2 = "#0a2135"
    PANEL_3 = "#0d2940"
    BORDER = "#123d5d"
    CYAN = "#19c8ff"
    BLUE = "#1785ff"
    GREEN = "#22e69b"
    PURPLE = "#655cff"
    TEXT = "#f2f7fb"
    MUTED = "#9bb6c9"
    DIM = "#5e7c91"
    RED = "#ff4b5c"
    YELLOW = "#ffd45a"
    TRAINING_IMAGE_RADIUS = 12

    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.configure(bg=self.BG)
        self.root.minsize(1050, 650)
        width, height = 1080, 720
        x = max(0, (self.root.winfo_screenwidth() - width) // 2)
        y = max(0, (self.root.winfo_screenheight() - height) // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

        self.camera = None
        self.camera_after = None
        self.camera_thread = None
        self.camera_lock = threading.Lock()
        self.camera_stop_event = threading.Event()
        self.camera_device_lock = threading.Lock()
        self.camera_frames = queue.Queue(maxsize=1)
        self.camera_mode = None
        self.cascade = None
        self.current_user_name = ""
        self.current_user_id = 0
        self.captured_count = 0
        self.stop_camera = False
        self.last_alert = 0
        self.training_thread = None
        self.training_queue = queue.Queue()
        self.training_running = False
        self.recognition_name = "Waiting..."
        self.recognition_conf = "--"
        self.recognition_status = "Ready"
        self.recognition_time = ""
        self.collection_message = "Position your face in the camera"

        self.build_shell()
        self.show_home()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # --------------------------- shell ---------------------------
    def build_shell(self):
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.page = tk.Frame(self.root, bg=self.BG)
        self.page.grid(row=0, column=0, sticky="nsew")
        self.page.grid_rowconfigure(0, weight=1)
        self.page.grid_columnconfigure(0, weight=1)

    def clear(self):
        self.stop_current_camera()
        for child in self.page.winfo_children():
            child.destroy()

    def header(self, parent, title=None, subtitle=None, icon="◉", status=None, status_color=None):
        bar = tk.Frame(parent, bg=self.HEADER, height=76)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        tk.Label(bar, text="◉", bg=self.HEADER, fg=self.CYAN,
                 font=("Segoe UI Symbol", 19, "bold")).pack(side="left", padx=(24, 8))
        tk.Label(bar, text="Face Recognition System", bg=self.HEADER, fg=self.TEXT,
                 font=("Segoe UI", 10, "bold")).pack(side="left")

        if status:
            pill = tk.Label(bar, text=f"●  {status}", bg="#082b2b" if status_color == self.GREEN else "#091f31",
                            fg=status_color or self.CYAN, padx=14, pady=6,
                            font=("Segoe UI", 8, "bold"))
            pill.pack(side="right", padx=22)

        if title:
            center = tk.Frame(bar, bg=self.HEADER)
            center.place(relx=0.5, rely=0.5, anchor="center")
            tk.Label(center, text=icon, bg=self.HEADER, fg=status_color or self.CYAN,
                     font=("Segoe UI Symbol", 23, "bold")).pack(side="left", padx=(0, 12))
            tb = tk.Frame(center, bg=self.HEADER)
            tb.pack(side="left")
            tk.Label(tb, text=title, bg=self.HEADER, fg=self.TEXT,
                     font=("Segoe UI", 19, "bold")).pack(anchor="w")
            tk.Label(tb, text=subtitle or "", bg=self.HEADER, fg=self.MUTED,
                     font=("Segoe UI", 8)).pack(anchor="w", pady=(1, 0))
        return bar

    def footer(self, parent, left="Smart Security Solution", right=None):
        f = tk.Frame(parent, bg=self.HEADER, height=48)
        f.pack(fill="x", side="bottom")
        f.pack_propagate(False)
        tk.Label(f, text=f"♙  {left}", bg=self.HEADER, fg=self.MUTED,
                 font=("Segoe UI", 8)).pack(side="left", padx=28)
        tk.Label(f, text="♢  Secure   |   Fast   |   Reliable", bg=self.HEADER, fg=self.MUTED,
                 font=("Segoe UI", 8)).pack(side="left", expand=True)
        tk.Label(f, text=right or datetime.now().strftime("%d %b %Y  %I:%M %p"),
                 bg=self.HEADER, fg=self.MUTED, font=("Segoe UI", 8)).pack(side="right", padx=28)

    # --------------------------- widgets ---------------------------
    def button(self, parent, text, command, bg=None, width=18, height=1):
        return RoundedButton(
            parent,
            text=text,
            command=command,
            width=max(190, width * 10),
            height=max(44, height * 32),
            radius=16,
            bg=bg or self.PANEL_3,
            hover_bg=self.BORDER,
            fg=self.TEXT,
            border=self.CYAN,
            border_width=2,
            font=("Segoe UI", 10, "bold"),
        )

    def lighten(self, color):
        return {
            self.CYAN: "#49d9ff", self.BLUE: "#4da5ff", self.GREEN: "#48efb0",
            self.PURPLE: "#817bff", self.RED: "#ff6b79", self.PANEL_3: "#153b58"
        }.get(color, "#1b4665")

    def card(self, parent, accent):
        return tk.Frame(parent, bg=self.PANEL, highlightbackground=accent,
                        highlightthickness=1, bd=0)

    def info_panel(self, parent, title, column=0, row=0, padding=(14, 12), centered=False, title_size=10):
        p = tk.Frame(parent, bg=self.BG, padx=padding[0], pady=padding[1])
        p.grid(row=row, column=column, sticky="nsew")
        self.rounded_panel_background(p, self.PANEL, self.BORDER)
        tk.Label(
            p,
            text=title,
            bg=self.PANEL,
            fg=self.TEXT,
            font=("Segoe UI", title_size, "bold")
        ).pack(
            fill="x",
            anchor="center" if centered else "w",
            pady=(2, 4)
        )
        tk.Frame(p, bg=self.BORDER, height=1).pack(fill="x", pady=(8, 8))
        return p

    def rounded_panel_background(self, panel, fill, outline):
        background = tk.Canvas(panel, bg=panel.master.cget("bg"), highlightthickness=0)
        background.place(relwidth=1, relheight=1)

        def draw(event):
            background.delete("all")
            self.rounded_rectangle(background, 1, 1, event.width - 1, event.height - 1,
                                   12, fill, outline)

        background.bind("<Configure>", draw)

    def info_row(
        self,
        parent,
        label,
        value,
        color=None,
        row_padding=5,
        centered=False,
        font_size=8,
        icon=None
    ):
        row = tk.Frame(parent, bg=self.PANEL)
        row.pack(fill="x", pady=row_padding, padx=8)

        # Fixed columns make Name, Confidence, Status and Date & Time
        # start at exactly the same position.
        content = tk.Frame(row, bg=self.PANEL)
        content.pack(anchor="center", padx=10, pady=5)

        icon_col = 0
        label_col = 1
        colon_col = 2
        value_col = 3

        if icon:
            tk.Label(
                content,
                text=icon,
                bg=self.PANEL,
                fg=self.CYAN,
                font=("Segoe UI Symbol", max(9, font_size + 1)),
                width=2,
                anchor="center"
            ).grid(row=0, column=icon_col, padx=(0, 8), sticky="e")
        else:
            tk.Label(
                content,
                text="",
                bg=self.PANEL,
                width=2
            ).grid(row=0, column=icon_col, padx=(0, 8))

        tk.Label(
            content,
            text=label,
            bg=self.PANEL,
            fg=self.MUTED,
            font=("Segoe UI", max(9, font_size - 1)),
            width=11,
            anchor="e"
        ).grid(row=0, column=label_col, sticky="e")

        tk.Label(
            content,
            text=":",
            bg=self.PANEL,
            fg=self.DIM,
            font=("Segoe UI", max(9, font_size - 1)),
            width=2,
            anchor="center"
        ).grid(row=0, column=colon_col, padx=5)

        value_label = tk.Label(
            content,
            text=value,
            bg=self.PANEL,
            fg=color or self.TEXT,
            font=("Segoe UI", max(10, font_size + 1), "bold"),
            width=18,
            anchor="w",
            justify="left"
        )
        value_label.grid(row=0, column=value_col, padx=(2, 0), sticky="w")

        return value_label

    # --------------------------- HOME ---------------------------
    def show_home(self):
        self.clear()
        root = tk.Frame(self.page, bg=self.BG)
        root.pack(fill="both", expand=True)

        hero = tk.Frame(root, bg=self.BG)
        hero.pack(fill="x", pady=(28, 18))
        hero_content = tk.Frame(hero, bg=self.BG)
        hero_content.pack(anchor="center")
        # Use the face-recognition logo cropped from the supplied reference image.
        self.home_logo_photo = None
        logo_candidates = [
            Path(__file__).resolve().parent / "venv" / "face_logo_from_reference.png",
            Path(__file__).resolve().parent / "face_logo_from_reference.png",
        ]
        for logo_path in logo_candidates:
            if logo_path.exists():
                logo_image = Image.open(logo_path).convert("RGBA")
                resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.ANTIALIAS
                logo_image.thumbnail((88, 88), resample)
                self.home_logo_photo = ImageTk.PhotoImage(logo_image)
                break

        if self.home_logo_photo is not None:
            tk.Label(
                hero_content,
                image=self.home_logo_photo,
                bg=self.BG
            ).pack(side="left", padx=(0, 18))
        else:
            self.home_face_mark(hero_content).pack(side="left", padx=(0, 18))
        title = tk.Frame(hero_content, bg=self.BG)
        title.pack(side="left")
        tk.Label(title, text="Face Recognition System", bg=self.BG, fg=self.TEXT,
                 font=("Segoe UI", 31, "bold")).pack(anchor="center")
        tk.Label(title, text="Face Detection & Recognition", bg=self.BG, fg="#80b9d8",
                 font=("Segoe UI", 14)).pack(anchor="center", pady=(4, 0))

        card_area = tk.Frame(root, bg=self.BG)
        card_area.pack(fill="both", expand=True)
        cards = tk.Frame(card_area, bg=self.BG)
        cards.place(relx=0.5, rely=0.5, anchor="center", width=960, height=260)
        for i in range(3):
            cards.grid_columnconfigure(i, weight=1)
        cards.grid_rowconfigure(0, weight=1, uniform="home-cards")

        # Reference-image icons for the three home cards and the Open arrow.
        asset_dir = Path(__file__).resolve().parent / "venv"
        self.home_icon_photos = []

        def load_home_icon(filename, size):
            path = asset_dir / filename
            if not path.exists():
                return None
            image = Image.open(path).convert("RGBA")
            resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.ANTIALIAS
            image.thumbnail(size, resample)
            photo = ImageTk.PhotoImage(image)
            self.home_icon_photos.append(photo)
            return photo

        camera_icon = load_home_icon("home_camera_icon.png", (52, 52))
        settings_icon = load_home_icon("home_settings_icon.png", (52, 52))
        recognition_icon = load_home_icon("home_recognition_icon.png", (58, 58))
        self.home_next_icon_photo = load_home_icon("next_arrow_icon.png", (11, 20))

        self.home_card(cards, 0, "📷", "Collect Faces",
                       "Add new user data and\ncapture face samples.", self.CYAN,
                       "Open", self.show_collect, icon_image=camera_icon)
        self.home_card(cards, 1, "⚙", "Train Recognizer",
                       "Train the model from\ncollected face datasets.", self.PURPLE,
                       "Open", self.show_train, icon_image=settings_icon)
        self.home_card(cards, 2, "♙⌕", "Real-Time Recognition",
                       "Run live detection and\ndisplay recognized identities.", self.GREEN,
                       "Open", self.show_recognition, icon_image=recognition_icon)

        self.footer(root)

    def home_face_mark(self, parent):
        mark = tk.Canvas(parent, width=84, height=84, bg=self.BG, highlightthickness=0)
        color = self.CYAN
        mark.create_line(9, 27, 9, 9, 27, 9, fill=color, width=3)
        mark.create_line(57, 9, 75, 9, 75, 27, fill=color, width=3)
        mark.create_line(9, 57, 9, 75, 27, 75, fill=color, width=3)
        mark.create_line(57, 75, 75, 75, 75, 57, fill=color, width=3)
        mark.create_oval(27, 17, 57, 60, outline=color, width=2)
        mark.create_arc(21, 43, 63, 78, start=0, extent=180, style="arc", outline=color, width=2)
        mark.create_oval(34, 33, 37, 36, fill=color, outline=color)
        mark.create_oval(47, 33, 50, 36, fill=color, outline=color)
        mark.create_line(42, 37, 40, 45, 44, 45, fill=color, width=1)
        mark.create_arc(36, 43, 49, 52, start=205, extent=130, style="arc", outline=color, width=1)
        return mark

    def home_card(self, parent, col, icon, title, description, accent, action_text, command, icon_image=None):
        card = tk.Canvas(parent, bg=self.BG, height=260, highlightthickness=0)
        card.grid(row=0, column=col, sticky="nsew", padx=8)

        # Home-screen Open button: rounded, wide, centered and matching
        # the reference design.
        action = tk.Canvas(
            card,
            bg=accent,
            height=58,
            highlightthickness=0,
            bd=0,
            cursor="hand2"
        )

        def draw_action(event=None, color=None):
            action.delete("all")
            width = max(action.winfo_width(), 1)
            height = max(action.winfo_height(), 1)
            fill = color if color else accent

            # Strong rounded shape, matching the reference image.
            self.rounded_rectangle(
                action, 1, 1, width - 1, height - 1,
                12, fill
            )

            # Center the complete Open + arrow group.
            center_x = width / 2

            action.create_text(
                center_x - 12,
                height / 2,
                text=action_text,
                fill="white",
                font=("Segoe UI", 11, "bold"),
                anchor="center"
            )

            if getattr(self, "home_next_icon_photo", None):
                action.create_image(
                    center_x + 58,
                    height / 2,
                    image=self.home_next_icon_photo,
                    anchor="center"
                )
            else:
                action.create_text(
                    center_x + 58,
                    height / 2,
                    text="›",
                    fill="white",
                    font=("Segoe UI", 18, "bold"),
                    anchor="center"
                )

        action.bind("<Configure>", draw_action)
        action.bind(
            "<Enter>",
            lambda event: draw_action(color=self.lighten(accent))
        )
        action.bind(
            "<Leave>",
            lambda event: draw_action(color=accent)
        )
        action.bind("<Button-1>", lambda event: command())

        def draw_card(event):
            card.delete("all")
            width = max(event.width, 1)
            height = max(event.height, 1)
            center = width / 2

            # Card border and inner panel.
            self.rounded_rectangle(
                card, 1, 1, width - 1, height - 1,
                14, accent
            )
            self.rounded_rectangle(
                card, 2, 2, width - 2, height - 2,
                13, self.PANEL
            )

            if icon_image is not None:
                card.create_image(
                    center, 48,
                    image=icon_image,
                    anchor="center"
                )
            else:
                card.create_text(
                    center, 48,
                    text=icon,
                    fill=accent,
                    font=(
                        "Segoe UI Emoji"
                        if icon == "📷"
                        else "Segoe UI Symbol",
                        34,
                        "bold"
                    )
                )

            card.create_text(
                center, 104,
                text=title,
                fill=self.TEXT,
                font=("Segoe UI", 15, "bold")
            )

            card.create_text(
                center, 151,
                text=description,
                fill="#b4c9d8",
                font=("Segoe UI", 9),
                justify="center",
                width=max(width - 36, 1)
            )

            # Exact reference-like Open button size.
            card.create_window(
                center,
                height - 50,
                window=action,
                width=min(width - 36, 238),
                height=58
            )

        card.bind("<Configure>", draw_card)

    def rounded_rectangle(self, canvas, x1, y1, x2, y2, radius, fill, outline=None):
        radius = min(radius, (x2 - x1) / 2, (y2 - y1) / 2)
        points = (
            x1 + radius, y1, x2 - radius, y1, x2, y1,
            x2, y1 + radius, x2, y2 - radius, x2, y2,
            x2 - radius, y2, x1 + radius, y2, x1, y2,
            x1, y2 - radius, x1, y1 + radius, x1, y1,
        )
        return canvas.create_polygon(points, smooth=True, splinesteps=16,
                         fill=fill, outline=outline or fill)

    # --------------------------- page header ---------------------------
    def page_top(self, root, title, subtitle, icon, color, status, icon_image=None, centered=False):
        bar = tk.Frame(root, bg=self.BG, height=86)
        bar.pack(fill="x", padx=22, pady=(12, 0))
        bar.pack_propagate(False)
        self.button(bar, "←  Back to Home", self.show_home, self.PANEL_3, 20, 1).pack(side="left", pady=10)

        mid = tk.Frame(bar, bg=self.BG)
        if centered:
            mid.place(relx=0.5, rely=0.5, anchor="center")
        else:
            mid.pack(side="left", padx=42)
        if icon_image is not None:
            tk.Label(mid, image=icon_image, bg=self.BG).pack(side="left", padx=(0, 12))
        else:
            tk.Label(mid, text=icon, bg=self.BG, fg=color,
                     font=("Segoe UI Symbol", 27, "bold")).pack(side="left", padx=(0, 12))
        tx = tk.Frame(mid, bg=self.BG)
        tx.pack(side="left")
        tk.Label(tx, text=title, bg=self.BG, fg=self.TEXT,
                 font=("Segoe UI", 20, "bold")).pack(anchor="w")
        tk.Label(tx, text=subtitle, bg=self.BG, fg=self.MUTED,
                 font=("Segoe UI", 8)).pack(anchor="w")

        pill = tk.Label(bar, text=f"●  {status}", bg="#062c2c" if color == self.GREEN else "#0a2031",
                        fg=color, padx=15, pady=7, font=("Segoe UI", 8, "bold"))
        pill.pack(side="right", pady=14)
        return bar

    # --------------------------- CAMERA PREVIEW ---------------------------
    def make_camera_panel(self, parent, title, subtitle, accent):
        panel = tk.Frame(parent, bg=self.PANEL, padx=8, pady=8)
        panel.grid(row=0, column=0, sticky="nsew")
        self.rounded_panel_background(panel, self.PANEL, accent)
        self.rounded_panel_background(panel, self.PANEL, accent)
        tk.Label(panel, text=title, bg=self.PANEL, fg=self.TEXT,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=6, pady=(0, 2))
        tk.Label(panel, text=subtitle, bg=self.PANEL, fg=accent,
                 font=("Segoe UI", 7, "bold")).pack(anchor="w", padx=6, pady=(0, 7))
        frame = tk.Frame(panel, bg="#01070d", highlightbackground="#164b68", highlightthickness=1)
        frame.pack(fill="both", expand=True)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        video = tk.Label(frame, bg="#01070d")
        video.grid(row=0, column=0, sticky="nsew")
        bottom = tk.Label(panel, text="●  Camera Active", bg="#062c2c", fg=self.GREEN,
                          anchor="w", padx=12, pady=7, font=("Segoe UI", 8, "bold"))
        bottom.pack(fill="x", pady=(7, 0))
        return video, bottom

    # --------------------------- COLLECT ---------------------------
    def show_collect(self):
        self.clear()
        root = tk.Frame(self.page, bg=self.BG)
        root.pack(fill="both", expand=True)
        self.collect_header_icon_photo = None
        collect_icon_path = Path(__file__).resolve().parent / "venv" / "home_camera_icon.png"
        if collect_icon_path.exists():
            collect_icon = Image.open(collect_icon_path).convert("RGBA")
            resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.ANTIALIAS
            collect_icon = collect_icon.resize((44, 44), resample)
            self.collect_header_icon_photo = ImageTk.PhotoImage(collect_icon)
        self.page_top(
            root,
            "Collect Faces",
            "Capture face samples for a new user",
            "◉",
            self.CYAN,
            "CAMERA READY",
            icon_image=self.collect_header_icon_photo,
            centered=True,
        )

        body = tk.Frame(root, bg=self.BG)
        body.pack(fill="both", expand=True, padx=24, pady=(0, 16))
        body.grid_columnconfigure(0, weight=5, uniform="collect-layout")
        body.grid_columnconfigure(1, weight=3, uniform="collect-layout")
        body.grid_rowconfigure(0, weight=1)

        video, status = self.make_camera_panel(body, "", "", self.CYAN)
        video.master.master.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.collect_video = video
        self.collect_status = status

        right = tk.Frame(body, bg=self.BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        user = self.info_panel(
            right,
            "USER INFORMATION",
            padding=(32, 26),
            centered=True,
            title_size=18
        )
        user.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        self.collect_name_var = tk.StringVar()
        self.collect_id_var = tk.StringVar()

        self.entry_row(user, "♙", "Name", self.collect_name_var)
        self.entry_row(user, "▣", "ID", self.collect_id_var)

        self.button(
            user,
            "▶  Start Collection",
            self.start_collection,
            self.BLUE,
            20,
            1
        ).pack(fill="x", pady=(16, 2))

        status_panel = self.info_panel(right, "♧  Collection Status")
        status_panel.grid(row=1, column=0, sticky="nsew")
        self.collect_count_label = self.info_row(status_panel, "Faces Captured", "0 / 500", self.TEXT)
        self.collect_state_label = self.info_row(status_panel, "Status", "Ready", self.GREEN)
        self.collect_message_label = self.info_row(status_panel, "Message", "Enter name and ID", self.TEXT)
        self.collect_camera_label = self.info_row(status_panel, "Camera", "Standby", self.CYAN)

    def entry_row(self, parent, icon, label, variable):
        row = tk.Frame(parent, bg=self.PANEL)
        row.pack(fill="x", pady=7, padx=10)

        # Keep the small details neatly aligned with comfortable
        # spacing from the User Information panel border.
        row.grid_columnconfigure(1, weight=1)

        # Icon + field name stay together as one compact group.
        label_group = tk.Frame(row, bg=self.PANEL)
        # Keep the text box close to the Name/ID label while retaining
        # comfortable spacing from the panel border.
        label_group.grid(row=0, column=0, padx=(4, 8), sticky="w")

        tk.Label(
            label_group,
            text=icon,
            bg=self.PANEL,
            fg=self.CYAN,
            font=("Segoe UI Symbol", 10, "bold")
        ).pack(side="left", padx=(0, 6))

        tk.Label(
            label_group,
            text=label,
            bg=self.PANEL,
            fg=self.MUTED,
            font=("Segoe UI", 9),
            anchor="w"
        ).pack(side="left")

        e = tk.Entry(
            row,
            textvariable=variable,
            bg="#0c2033",
            fg=self.TEXT,
            insertbackground=self.TEXT,
            relief="flat",
            font=("Segoe UI", 9)
        )
        e.grid(row=0, column=1, padx=(0, 4), ipady=6, sticky="ew")

    def start_collection(self):
        name = self.collect_name_var.get().strip()
        user_id_text = self.collect_id_var.get().strip()
        if not name:
            messagebox.showerror("Invalid Name", "Please enter a name.", parent=self.root)
            return

        # Name must start with a letter. After the first letter,
        # letters, numbers, and spaces are allowed.
        # Examples accepted: Chandan, chandn12, Chandan123.
        # Examples rejected: 123Chandan, _Chandan, @Chandan, %Chandan.
        if not name[0].isalpha() or not all(ch.isalnum() or ch.isspace() for ch in name):
            messagebox.showerror(
                "Invalid Name",
                "Invalid name. Name must start with a letter. Letters, numbers, and spaces are allowed after that.",
                parent=self.root
            )
            return

        try:
            user_id = int(user_id_text)
        except ValueError:
            messagebox.showerror("Invalid ID", "Please enter a numeric ID.", parent=self.root)
            return

        if self.camera_mode:
            return
        self.current_user_name = name
        self.current_user_id = user_id
        self.captured_count = 0
        os.makedirs(os.path.join(DATASET_PATH, f"{name}_{user_id}"), exist_ok=True)
        self.start_camera("collect")

    # --------------------------- TRAIN ---------------------------
    def show_train(self):
        self.clear()
        root = tk.Frame(self.page, bg=self.BG)
        root.pack(fill="both", expand=True)
        gear_path = Path(__file__).resolve().parent / "venv" / "home_settings_icon.png"
        self.training_gear_photo = None
        if gear_path.exists():
            gear_image = Image.open(gear_path).convert("RGBA")
            resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.ANTIALIAS
            gear_image = gear_image.resize((44, 44), resample)
            self.training_gear_photo = ImageTk.PhotoImage(gear_image)
        self.page_top(
            root,
            "Train Recognizer",
            "Train the model from collected face datasets",
            "⚙",
            self.PURPLE,
            "TRAINING READY",
            icon_image=self.training_gear_photo,
            centered=True,
        )

        body = tk.Frame(root, bg=self.BG)
        body.pack(fill="both", expand=True, padx=24, pady=(0, 16))
        body.grid_columnconfigure(0, weight=5)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=self.PANEL, highlightbackground=self.BORDER, highlightthickness=1, padx=18, pady=15)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        tk.Label(left, text="Training Recognizer...", bg=self.PANEL, fg=self.TEXT,
             font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(2, 0))
        tk.Label(left, text="Please wait while the model is being trained.", bg=self.PANEL, fg=self.MUTED,
             font=("Segoe UI", 8)).pack(anchor="w", pady=(2, 0))

        progress_area = tk.Frame(left, bg=self.PANEL)
        progress_area.pack(fill="x", pady=(2, 5))
        self.progress_canvas = tk.Canvas(progress_area, width=150, height=150, bg=self.PANEL, highlightthickness=0)
        self.progress_canvas.pack(anchor="center")
        self.progress_bg = self.progress_canvas.create_oval(13, 13, 137, 137, outline="#18384f", width=10)
        self.progress_arc = self.progress_canvas.create_arc(13, 13, 137, 137, start=90, extent=1,
                                    outline=self.BLUE, width=10, style="arc")
        self.progress_text = self.progress_canvas.create_text(75, 75, text="0%", fill=self.TEXT,
                                       font=("Segoe UI", 17, "bold"))
        self.training_status_text = tk.StringVar(value="Ready to train your recognizer")
        tk.Label(progress_area, textvariable=self.training_status_text, bg=self.PANEL, fg=self.TEXT,
             font=("Segoe UI", 11, "bold")).pack(pady=(0, 3))
        tk.Label(progress_area, text="Please wait while the model is being trained.", bg=self.PANEL,
             fg=self.MUTED, font=("Segoe UI", 8)).pack()

        steps_box = tk.Frame(left, bg="#061422", highlightbackground=self.BORDER, highlightthickness=1, padx=14, pady=8)
        steps_box.pack(fill="both", expand=True, pady=(5, 0))
        self.step_labels = []
        for text in ["Loading face datasets...", "Preparing training data...", "Training model...", "Saving trained model...", "Training completed successfully!"]:
            lbl = tk.Label(steps_box, text="○  " + text, bg="#061422", fg=self.MUTED,
                           anchor="w", font=("Segoe UI", 8))
            lbl.pack(fill="x", pady=4)
            self.step_labels.append(lbl)

        self.train_button = self.button(left, "⚙  Start Training", self.start_training, self.PURPLE, 22, 1)
        self.train_button.pack(fill="x", pady=(10, 0))

        right = tk.Frame(body, bg=self.BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)
        art = tk.Canvas(right, bg=self.BG, height=180, highlightthickness=0, bd=0)
        art.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.draw_training_face(art)

        details = self.info_panel(
            right,
            "♧  Training Information",
            padding=(26, 22),
            centered=True,
            title_size=11,
        )
        details.grid(row=1, column=0, sticky="nsew")
        self.train_status = self.info_row(details, "Status", "Ready", self.GREEN, 8, centered=True, font_size=9)
        self.train_images = self.info_row(details, "Total Images", self.dataset_image_count(), self.TEXT, 8, centered=True, font_size=9)
        self.train_users = self.info_row(details, "Total Users", self.dataset_user_count(), self.TEXT, 8, centered=True, font_size=9)
        self.info_row(details, "Model Type", "LBPH", self.TEXT, 8, centered=True, font_size=9)
        self.info_row(details, "Model File", "Trainer.yml", self.TEXT, 8, centered=True, font_size=9)

    def draw_training_face(self, canvas):
        image_path = Path(__file__).resolve().parent / "venv" / "train_face_photo.png"
        self.training_face_source = None
        if image_path.exists():
            self.training_face_source = Image.open(image_path).convert("RGB")

        color = "#087db5"
        canvas.bind(
            "<Configure>",
            lambda event: self.render_training_face(canvas, event.width, event.height, color),
        )

    def render_training_face(self, canvas, width, height, color):
        if self.training_face_source is None:
            self._draw_training_face(canvas, width, height, color)
            return

        scale = min(width / self.training_face_source.width, height / self.training_face_source.height)
        image_size = (
            max(1, int(self.training_face_source.width * scale)),
            max(1, int(self.training_face_source.height * scale)),
        )
        resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.ANTIALIAS
        image = self.training_face_source.resize(image_size, resample).convert("RGBA")
        mask = Image.new("L", image_size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle(
            (0, 0, image_size[0] - 1, image_size[1] - 1),
            radius=min(self.TRAINING_IMAGE_RADIUS, image_size[0] // 2, image_size[1] // 2),
            fill=255,
        )
        image.putalpha(mask)
        photo = ImageTk.PhotoImage(image)
        canvas.delete("all")
        canvas.create_image(width // 2, height // 2, image=photo, anchor="center")
        canvas.image = photo

    def _draw_training_face(self, canvas, width, height, color):
        canvas.delete("all")
        center_x = width / 2
        center_y = height / 2
        face_height = min(height * 0.82, width * 0.72)
        face_width = face_height * 0.76
        top = center_y - face_height * 0.49
        bottom = center_y + face_height * 0.51
        mesh = "#07517b"
        mesh_bright = "#0a78aa"

        for x in range(0, width + 1, 24):
            canvas.create_line(x, 0, x, height, fill="#061c2d")
        for y in range(0, height + 1, 24):
            canvas.create_line(0, y, width, y, fill="#061c2d")

        shape = ((0.00, 0.18), (0.08, 0.34), (0.22, 0.45), (0.45, 0.49),
                 (0.66, 0.44), (0.84, 0.33), (0.94, 0.20), (1.00, 0.06))

        def half_width(position):
            for index in range(len(shape) - 1):
                start_y, start_width = shape[index]
                end_y, end_width = shape[index + 1]
                if position <= end_y:
                    portion = (position - start_y) / (end_y - start_y)
                    return face_width * (start_width + (end_width - start_width) * portion)
            return face_width * shape[-1][1]

        def face_point(horizontal, vertical):
            y = top + vertical * face_height
            width_at_y = half_width(vertical)
            x = center_x + horizontal * width_at_y
            x += math.sin(vertical * math.pi) * horizontal * face_width * 0.045
            return x, y

        vertical_steps = 36
        for line_index in range(-9, 10):
            horizontal = line_index / 10
            points = []
            for step in range(vertical_steps + 1):
                point = face_point(horizontal, step / vertical_steps)
                points.extend(point)
            line_color = mesh_bright if line_index % 3 == 0 else mesh
            canvas.create_line(*points, fill=line_color, width=1, smooth=True)

        for row in range(1, 12):
            vertical = row / 12
            points = []
            for step in range(33):
                horizontal = -0.98 + step * 0.06125
                x, y = face_point(horizontal, vertical)
                y += math.cos(horizontal * math.pi / 2) * 2
                points.extend((x, y))
            canvas.create_line(*points, fill=mesh_bright if row % 3 == 0 else mesh,
                               width=1, smooth=True)

        for row in range(2, 11, 2):
            vertical = row / 12
            for column in range(-8, 9, 2):
                x, y = face_point(column / 10, vertical)
                canvas.create_oval(x - 1, y - 1, x + 1, y + 1, fill="#18a4d6", outline="")

        eye_y = top + face_height * 0.43
        for eye_x in (center_x - face_width * 0.2, center_x + face_width * 0.2):
            canvas.create_line(eye_x - 12, eye_y, eye_x - 4, eye_y - 3,
                               eye_x + 5, eye_y - 3, eye_x + 12, eye_y,
                               fill="#26b9e7", width=2, smooth=True)
            canvas.create_oval(eye_x - 2, eye_y - 3, eye_x + 2, eye_y + 1,
                               fill="#48d8ff", outline="")

        nose_y = top + face_height * 0.47
        canvas.create_line(center_x, nose_y, center_x - 3, nose_y + face_height * 0.16,
                           center_x + 5, nose_y + face_height * 0.19,
                           fill="#1593c4", width=2, smooth=True)
        mouth_y = top + face_height * 0.76
        canvas.create_line(center_x - 14, mouth_y, center_x - 6, mouth_y + 3,
                           center_x + 5, mouth_y + 3, center_x + 14, mouth_y,
                           fill="#23a9d6", width=2, smooth=True)

        left = center_x - face_width * 0.58
        right = center_x + face_width * 0.58
        bracket_top = top - 5
        bracket_bottom = bottom + 5
        corner = 20
        canvas.create_line(left, bracket_top + corner, left, bracket_top, left + corner, bracket_top,
                           fill=self.CYAN, width=2)
        canvas.create_line(right - corner, bracket_top, right, bracket_top, right, bracket_top + corner,
                           fill=self.CYAN, width=2)
        canvas.create_line(left, bracket_bottom - corner, left, bracket_bottom, left + corner, bracket_bottom,
                           fill=self.CYAN, width=2)
        canvas.create_line(right - corner, bracket_bottom, right, bracket_bottom,
                           right, bracket_bottom - corner, fill=self.CYAN, width=2)

    def dataset_image_count(self):
        count = 0
        if os.path.exists(DATASET_PATH):
            for root, _, files in os.walk(DATASET_PATH):
                count += sum(1 for f in files if f.lower().endswith((".jpg", ".jpeg", ".png")))
        return str(count)

    def dataset_user_count(self):
        if not os.path.exists(DATASET_PATH):
            return "0"
        users = 0
        for name in os.listdir(DATASET_PATH):
            if os.path.isdir(os.path.join(DATASET_PATH, name)):
                users += 1
        return str(users)

    def start_training(self):
        if self.training_running:
            return
        self.training_running = True
        if hasattr(self, "training_status_text"):
            self.training_status_text.set("Loading face datasets...")
        self.train_button.configure(state="disabled", text="⚙  Training in progress...", bg="#244563")
        self.train_status.configure(text="Training...", fg=self.YELLOW)
        for lbl in self.step_labels:
            lbl.configure(text="○  " + lbl.cget("text").split("  ", 1)[-1], fg=self.MUTED)
        self.training_thread = threading.Thread(target=self.training_worker, daemon=True)
        self.training_thread.start()
        self.root.after(100, self.poll_training)

    def training_worker(self):
        def callback(percent, text):
            self.training_queue.put((percent, text))
        ok = train_recognizer(callback)
        self.training_queue.put(("done", ok))

    def poll_training(self):
        try:
            while True:
                item = self.training_queue.get_nowait()
                if item[0] == "done":
                    ok = item[1]
                    self.training_running = False
                    self.train_button.configure(state="normal", text="⚙  Start Training", bg=self.PURPLE)
                    self.train_status.configure(text="Ready" if ok else "Failed", fg=self.GREEN if ok else self.RED)
                    if ok and hasattr(self, "recognizer"):
                        self.recognizer = None
                    if hasattr(self, "training_status_text"):
                        self.training_status_text.set(
                            "Training completed successfully!" if ok else "Training failed. Check the dataset."
                        )
                    if ok:
                        messagebox.showinfo("Training", "Training completed successfully!", parent=self.root)
                    else:
                        messagebox.showerror("Training", "Training could not be completed. Check the dataset.", parent=self.root)
                    return
                percent, text = item
                self.update_training_ui(percent, text)
        except queue.Empty:
            pass
        if self.training_running:
            self.root.after(100, self.poll_training)

    def update_training_ui(self, percent, text):
        extent = max(1, min(359, int(percent * 3.59)))
        self.progress_canvas.itemconfigure(self.progress_arc, extent=-extent)
        self.progress_canvas.itemconfigure(self.progress_text, text=f"{percent}%")
        if hasattr(self, "training_status_text"):
            self.training_status_text.set(text)
        active_index = 0
        if "Preparing" in text:
            active_index = 1
        elif "Training model" in text:
            active_index = 2
        elif "Saving" in text:
            active_index = 3
        elif "completed" in text:
            active_index = 4
        for i, lbl in enumerate(self.step_labels):
            if i < active_index:
                lbl.configure(text="✓  " + lbl.cget("text").split("  ", 1)[-1], fg=self.GREEN)
            elif i == active_index:
                lbl.configure(text="●  " + lbl.cget("text").split("  ", 1)[-1], fg=self.CYAN)
            else:
                lbl.configure(text="○  " + lbl.cget("text").split("  ", 1)[-1], fg=self.MUTED)

    # --------------------------- RECOGNITION ---------------------------
    def show_recognition(self):
        self.clear()
        root = tk.Frame(self.page, bg=self.BG)
        root.pack(fill="both", expand=True)
        logo_path = Path(__file__).resolve().parent / "venv" / "real_time_recognition_logo.png"
        self.recognition_logo_photo = None
        if logo_path.exists():
            logo_image = Image.open(logo_path).convert("RGBA")
            resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.ANTIALIAS
            logo_image = logo_image.resize((44, 44), resample)
            self.recognition_logo_photo = ImageTk.PhotoImage(logo_image)
        header = self.page_top(
            root,
            "Real-Time Recognition",
            "Detect and recognize faces in real time",
            "♙⌕",
            self.GREEN,
            "RECOGNITION READY",
            icon_image=self.recognition_logo_photo,
            centered=True,
        )
        self.recognition_header_status = header.winfo_children()[-1]

        body = tk.Frame(root, bg=self.BG)
        body.pack(fill="both", expand=True, padx=24, pady=(0, 16))
        body.grid_columnconfigure(0, weight=5, uniform="recognition-layout")
        body.grid_columnconfigure(1, weight=3, uniform="recognition-layout")
        body.grid_rowconfigure(0, weight=1)

        video, status = self.make_camera_panel(body, "", "", self.GREEN)
        video.master.master.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.recognition_video = video
        self.recognition_status_bar = status
        status.configure(text="●  Recognition ready", fg=self.MUTED)

        right = tk.Frame(body, bg=self.BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        details = self.info_panel(
            right,
            "Recognition Details",
            padding=(36, 28),
            centered=True,
            title_size=18
        )
        details.grid(row=0, column=0, sticky="nsew")

        self.rec_name = self.info_row(
            details, "Name", "Waiting...",
            self.GREEN,
            row_padding=8,
            font_size=11,
            icon="♙",
            centered=True
        )
        self.add_info_divider(details)

        self.rec_conf = self.info_row(
            details, "Confidence", "--",
            self.TEXT,
            row_padding=8,
            font_size=10,
            icon="▥",
            centered=True
        )
        self.add_info_divider(details)

        self.rec_status = self.info_row(
            details, "Status", "Ready",
            self.GREEN,
            row_padding=8,
            font_size=10,
            icon="✓",
            centered=True
        )
        self.add_info_divider(details)

        self.rec_time = self.info_row(
            details, "Date & Time",
            datetime.now().strftime("%d %b %Y %I:%M %p"),
            self.TEXT,
            row_padding=8,
            font_size=9,
            icon="◷",
            centered=True
        )

        self.recognition_start_button = self.button(
            details,
            "▶  Start Recognition",
            self.start_recognition,
            self.GREEN,
            22,
            2
        )
        self.recognition_start_button.pack(fill="x", pady=(22, 10))

        self.button(
            details,
            "■  Stop Camera",
            self.stop_current_camera,
            self.RED,
            22,
            2
        ).pack(fill="x")

    def add_info_divider(self, parent):
        tk.Frame(parent, bg=self.BORDER, height=1).pack(fill="x", pady=(1, 3))

    def start_recognition(self):
        if self.camera_mode == "recognition":
            self.stop_current_camera()
            return
        if self.camera_mode:
            return
        self.start_camera("recognition")

    # --------------------------- camera engine ---------------------------
    def start_camera(self, mode):
        self.stop_current_camera()
        self.camera_mode = mode
        self.stop_camera = False
        self.captured_count = 0
        self.collection_message = "Position your face in the camera"
        self.recognition_name = "Waiting..."
        self.recognition_conf = "--"
        self.recognition_status = "Ready"
        stop_event = threading.Event()
        frame_queue = queue.Queue(maxsize=1)
        self.camera_stop_event = stop_event
        self.camera_frames = frame_queue
        status_bar = self.collect_status if mode == "collect" else self.recognition_status_bar
        status_bar.configure(text="●  Opening camera...", fg=self.CYAN)
        if mode == "recognition":
            self.recognition_header_status.configure(
                text="●  RECOGNITION CONNECTING", bg="#0a2031", fg=self.CYAN,
            )
            self.recognition_start_button.configure(text="■  Stop Recognition", bg=self.RED)
        self.camera_thread = threading.Thread(
            target=self.camera_worker,
            args=(mode, stop_event, frame_queue),
            daemon=True,
        )
        self.camera_thread.start()
        self.camera_after = self.root.after(20, self.update_camera)

    def camera_worker(self, mode, stop_event, frame_queue):
        with self.camera_device_lock:
            if stop_event.is_set():
                return
            video = cv2.VideoCapture(0)
            if not video.isOpened():
                video.release()
                self.publish_camera_update(frame_queue, ("error", "Could not access the camera."))
                return

            try:
                video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cascade = get_face_cascade()
                model_queue = None
                if (mode == "recognition" and getattr(self, "recognizer", None) is None
                    and os.path.exists(MODEL_PATH) and os.path.exists(ID_TO_NAME_FILE)
                    and int(self.dataset_image_count()) > 0):
                    model_queue = queue.Queue(maxsize=1)
                    threading.Thread(
                        target=self.load_recognition_model,
                        args=(model_queue,),
                        daemon=True,
                    ).start()

                while not stop_event.is_set():
                    ok, frame = video.read()
                    if not ok:
                        self.publish_camera_update(frame_queue, ("error", "Could not read from the camera."))
                        break

                    if mode == "recognition" and model_queue is not None:
                        try:
                            model_result = model_queue.get_nowait()
                        except queue.Empty:
                            model_result = None
                        if model_result is not None:
                            if model_result[0] == "error":
                                self.publish_camera_update(frame_queue, ("error", model_result[1]))
                                break
                            self.recognizer, self.name_list = model_result[1], model_result[2]
                            model_queue = None

                        if model_queue is not None:
                            self.publish_camera_update(
                                frame_queue,
                                (
                                    "frame", frame.copy(), self.captured_count, self.collection_message,
                                    "Waiting...", "--", "Loading model", datetime.now().strftime("%d %b %Y %I:%M %p"),
                                ),
                            )
                            continue

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.cascade.detectMultiScale(gray, 1.3, 5)
                    if mode == "collect":
                        self.process_collection_frame(frame, gray, faces)
                    else:
                        self.process_recognition_frame(frame, gray, faces)

                    update = (
                        "frame", frame, self.captured_count, self.collection_message,
                        self.recognition_name, self.recognition_conf,
                        self.recognition_status, self.recognition_time,
                    )
                    self.publish_camera_update(frame_queue, update)
            except Exception as error:
                self.publish_camera_update(frame_queue, ("error", str(error)))
            finally:
                video.release()

    def publish_camera_update(self, frame_queue, update):
        try:
            frame_queue.put_nowait(update)
        except queue.Full:
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                frame_queue.put_nowait(update)
            except queue.Full:
                pass

    def update_camera(self):
        self.camera_after = None
        mode = self.camera_mode
        if self.stop_camera or mode is None:
            return
        try:
            update = self.camera_frames.get_nowait()
        except queue.Empty:
            update = None

        if update:
            if update[0] == "error":
                error = update[1]
                self.stop_current_camera()
                status_bar = self.collect_status if mode == "collect" else self.recognition_status_bar
                status_bar.configure(text="●  Camera unavailable", fg=self.RED)
                messagebox.showerror("Camera Error", error, parent=self.root)
                return

            _, frame, count, message, name, confidence, status, timestamp = update
            widget = self.collect_video if mode == "collect" else self.recognition_video
            self.display_frame(frame, widget)
            if mode == "collect":
                self.collect_count_label.configure(text=f"{min(count, 500)} / 500")
                self.collect_state_label.configure(
                    text="Collecting..." if count < 500 else "Completed",
                    fg=self.CYAN if count < 500 else self.GREEN,
                )
                self.collect_message_label.configure(
                    text=message,
                    fg=self.TEXT if message not in ("Face not detected", "Collection completed") else self.YELLOW,
                )
                self.collect_camera_label.configure(text="Active", fg=self.GREEN)
                self.collect_status.configure(
                    text=f"●  {message}",
                    fg=self.YELLOW if message == "Face not detected" else self.GREEN,
                )
            else:
                self.recognition_header_status.configure(
                    text="●  RECOGNITION ACTIVE", bg="#062c2c", fg=self.GREEN,
                )
                self.rec_name.configure(text=name, fg=self.GREEN if status == "Recognized" else self.RED if status == "Unknown" else self.TEXT)
                self.rec_conf.configure(text=confidence, fg=self.TEXT)
                status_color = self.CYAN if status == "Loading model" else self.GREEN if status == "Recognized" else self.RED if status == "Unknown" else self.YELLOW
                self.rec_status.configure(text=status, fg=status_color)
                self.rec_time.configure(text=timestamp)
                if status == "Loading model":
                    self.recognition_status_bar.configure(text="●  Loading recognition model...", fg=self.CYAN)
                else:
                    self.recognition_status_bar.configure(
                        text="●  Recognizing faces..." if status in ("Recognized", "Unknown") else "●  Looking for faces...",
                        fg=self.GREEN if status in ("Recognized", "Unknown") else self.YELLOW,
                    )

        self.camera_after = self.root.after(30, self.update_camera)

    def process_collection_frame(self, frame, gray, faces):
        self.collection_message = "Face not detected" if len(faces) == 0 else "Keep looking at the camera"
        for (x, y, w, h) in faces:
            self.draw_corner_box(frame, x, y, w, h, (0, 255, 120), 3, 24)
            if self.captured_count < 500:
                self.captured_count += 1
                face_image = gray[y:y+h, x:x+w]
                user_dir = os.path.join(DATASET_PATH, f"{self.current_user_name}_{self.current_user_id}")
                os.makedirs(user_dir, exist_ok=True)
                path = os.path.join(user_dir, f"{self.current_user_name}.{self.captured_count}.jpg")
                cv2.imwrite(path, face_image)
                if self.captured_count >= 500:
                    self.collection_message = "Collection completed"
            break


    def process_recognition_frame(self, frame, gray, faces):
        if getattr(self, "recognizer", None) is None:
            if len(faces) == 0:
                self.recognition_name = "Waiting..."
                self.recognition_conf = "--"
                self.recognition_status = "No face"
            else:
                for (x, y, w, h) in faces:
                    self.draw_corner_box(frame, x, y, w, h, (0, 0, 255), 3, 24)
                    banner_h = 38
                    y1 = max(0, y - banner_h)
                    cv2.rectangle(frame, (x, y1), (x + w, y), (0, 0, 255), -1)
                    cv2.putText(frame, "Unknown", (x + 10, max(18, y - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
                    break
                self.recognition_name = "Unknown"
                self.recognition_conf = "--"
                self.recognition_status = "Unknown"
                self.alert_unknown_face()
            self.recognition_time = datetime.now().strftime("%d %b %Y %I:%M %p")
            return

        best = None
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            serial, conf = self.recognizer.predict(roi)
            known = 0 <= serial < len(self.name_list) and conf < CONFIDENCE_THRESHOLD
            label = self.name_list[serial] if known else "Unknown"
            color = (0, 255, 120) if known else (0, 0, 255)
            self.draw_corner_box(frame, x, y, w, h, color, 3, 24)
            banner_h = 38
            y1 = max(0, y - banner_h)
            cv2.rectangle(frame, (x, y1), (x + w, y), color, -1)
            cv2.putText(frame, f"{label}  ({max(0, int(100-conf))}%)", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            if best is None or conf < best[1]:
                best = (label, conf, known)
            if not known:
                self.alert_unknown_face()
            break

        if best:
            name, conf, known = best
            self.recognition_name = name
            self.recognition_conf = f"{max(0, int(100-conf))}%"
            self.recognition_status = "Recognized" if known else "Unknown"
        else:
            self.recognition_name = "Waiting..."
            self.recognition_conf = "--"
            self.recognition_status = "No face"
        self.recognition_time = datetime.now().strftime("%d %b %Y %I:%M %p")

    def alert_unknown_face(self):
        current_time = datetime.now().timestamp()
        if current_time - self.last_alert > 3:
            self.last_alert = current_time
            threading.Thread(target=play_alert_sound, daemon=True).start()

    def load_recognition_model(self, result_queue):
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(MODEL_PATH)
            with open(ID_TO_NAME_FILE, "r") as name_file:
                user_to_id = json.load(name_file)
            name_list = ["Unknown"] * (max(user_to_id.values()) + 1)
            for name, user_id in user_to_id.items():
                name_list[user_id] = name
            result_queue.put(("ok", recognizer, name_list))
        except Exception as error:
            result_queue.put(("error", str(error)))

    def draw_corner_box(self, frame, x, y, w, h, color, thickness=3, corner=22):
        # Four-corner scanner box, matching the supplied reference more closely than a full rectangle.
        cv2.line(frame, (x, y), (x + corner, y), color, thickness)
        cv2.line(frame, (x, y), (x, y + corner), color, thickness)
        cv2.line(frame, (x+w-corner, y), (x+w, y), color, thickness)
        cv2.line(frame, (x+w, y), (x+w, y+corner), color, thickness)
        cv2.line(frame, (x, y+h-corner), (x, y+h), color, thickness)
        cv2.line(frame, (x, y+h), (x+corner, y+h), color, thickness)
        cv2.line(frame, (x+w-corner, y+h), (x+w, y+h), color, thickness)
        cv2.line(frame, (x+w, y+h-corner), (x+w, y+h), color, thickness)

    def display_frame(self, frame, widget):
        if widget is None or not widget.winfo_exists():
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        width = max(widget.winfo_width(), 300)
        height = max(widget.winfo_height(), 200)
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, height), (1, 7, 13))
        x = (width - image.width) // 2
        y = (height - image.height) // 2
        canvas.paste(image, (x, y))
        photo = ImageTk.PhotoImage(canvas)
        widget.configure(image=photo)
        widget.image = photo

    def stop_current_camera(self):
        previous_mode = self.camera_mode
        self.stop_camera = True
        self.camera_stop_event.set()
        if self.camera_after:
            try:
                self.root.after_cancel(self.camera_after)
            except Exception:
                pass
            self.camera_after = None
        self.camera = None
        self.camera_mode = None
        if self.camera_thread and self.camera_thread.is_alive() and self.camera_thread is not threading.current_thread():
            self.camera_thread.join(timeout=1.0)
        self.camera_thread = None
        if hasattr(self, "recognizer"):
            self.recognizer = None
        if previous_mode == "recognition" and hasattr(self, "recognition_header_status"):
            self.recognition_header_status.configure(
                text="●  RECOGNITION READY", bg="#062c2c", fg=self.GREEN,
            )
            if hasattr(self, "recognition_status_bar"):
                self.recognition_status_bar.configure(text="●  Recognition stopped", fg=self.MUTED)
            if hasattr(self, "recognition_start_button"):
                self.recognition_start_button.configure(text="▶  Start Recognition", bg=self.GREEN)

    def on_close(self):
        self.stop_current_camera()
        self.root.destroy()


if __name__ == "__main__":
    instance_lock = acquire_single_instance_lock()
    if instance_lock is None:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Already Running", "Face Recognition System is already open.", parent=root)
        root.destroy()
    else:
        root = tk.Tk()
        app = FaceRecognitionApp(root)
        try:
            root.mainloop()
        finally:
            msvcrt.locking(instance_lock.fileno(), msvcrt.LK_UNLCK, 1)
            instance_lock.close()
