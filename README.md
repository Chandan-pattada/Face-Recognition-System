# 🧠 Face Recognition System using OpenCV and Tkinter

This is a complete GUI-based Face Recognition System built using Python, OpenCV, and Tkinter. The system allows users to:

- 📸 Collect face datasets from a webcam
- 🏷️ Train a face recognizer model using LBPH
- 🎯 Recognize faces in real-time
- 🔔 Alert when an unknown face is detected

---

## 📂 Project Structure

```
Face-Recognition-System/
│
├── assets/
│   ├── branding/
│   │   └── face_logo_from_reference.png
│   │
│   ├── home/
│   │   ├── home_camera_icon.png
│   │   ├── home_recognition_icon.png
│   │   ├── home_settings_icon.png
│   │   └── next_arrow_icon.png
│   │
│   ├── recognition/
│   │   └── real_time_recognition_logo.png
│   │
│   └── training/
│       ├── settings_gear_image.png
│       └── train_face_photo.png
│
├── data/
│   └── id_to_name.json
│
├── sample/
│   └── 5.jpg
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```
### Generated Files

The following files and folders are generated locally while using the application and are excluded from the GitHub repository:

- `datasets/` – Stores collected face images organized by user.
- `Trainer.yml` – Stores the trained LBPH face recognition model.
- `.venv/` – Python virtual environment.
- `venv/` – Additional local Python virtual environment.
- `__pycache__/` – Python-generated cache files.
---

## ⚙️ Features

- **Face Detection**: Uses Haar Cascade Classifier
- **Face Recognition**: Uses OpenCV’s LBPH algorithm
- **GUI**: Built using Tkinter for user-friendly interaction
- **Image Augmentation**: Rotations and flips for better training
- **Real-time Monitoring**: Webcam-based recognition with live feedback
- **Sound Alert**: Beep sound when unknown face is detected

---

## 🚀 Requirements

Install the required Python libraries:

```bash
pip install opencv-python opencv-contrib-python pillow numpy
```

---

## ▶️ How to Run

1. **Activate virtual environment** (if using one):
   ```bash
   .\venv\Scripts\activate   # For Windows
   ```

2. **Run the main script**:
   ```bash
   python main.py
   ```

---
## 🧠 How It Works

1. **Collect Faces**
   - Enter the user's name and ID.
   - Captures face images using the webcam.
   - Stores the collected face images in the `datasets/` directory.
   - Captures up to 500 face samples for each user.

2. **Train Recognizer**
   - Uses the collected face dataset to train the LBPH face recognizer.
   - Generates the trained recognition model as `Trainer.yml`.
   - Stores the ID-to-name mapping in `data/id_to_name.json`.

3. **Real-Time Recognition**
   - Accesses the webcam for real-time face detection and recognition.
   - Uses the trained LBPH model to identify registered users.
   - Displays the recognized user's name and confidence.
   - Displays `Unknown` when a face cannot be recognized.

---

## 🛠️ Future Improvements

- Improve recognition accuracy under different lighting conditions.
- Add recognition history and access logs.
- Store recognition records in a database.
- Add additional security and authentication features.
- Develop a web-based version of the system.
- Improve face detection and recognition performance.