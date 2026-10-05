# 🧠 Face Recognition System using OpenCV and Tkinter

This is a complete GUI-based Face Recognition System built using Python, OpenCV, and Tkinter. The system allows users to:

- 📸 Collect face datasets from a webcam
- 🏷️ Train a face recognizer model using LBPH
- 🎯 Recognize faces in real-time
- 🔔 Alert when an unknown face is detected

---

## 📂 Project Structure

```
face_recognition/
│
├── datasets/               # Collected face images organized by user
├── main.py                 # Main Python script with GUI
├── Trainer.yml             # Trained LBPH face recognition model (generated after training)
├── id_to_name.json         # Mapping of user IDs to names
├── image.png               # Background image for GUI (replace as needed)
└── README.md               # This file
```

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

1. **Collect Dataset**:
   - Prompts for user ID and name
   - Captures face images from webcam and stores them in `/datasets`

2. **Train Recognizer**:
   - Trains the LBPH recognizer on the dataset
   - Saves the model and ID-name mapping

3. **Real-Time Recognition**:
   - Recognizes faces from webcam input
   - Displays name if recognized, otherwise shows "Unknown" and plays alert

---

## 📸 Screenshot

![Face Recognition GUI](preview.png) <!-- Replace with actual screenshot if available -->

---

## 🛠️ Future Improvements

- Add face mask detection
- Store recognition logs in a database
- Deploy with Flask or Streamlit for web interface
- Add email notification for unauthorized access

---

## 👨‍💻 Developed By

**Abdul Hazeez**  
[Computer Science Engineering | KVG College of Engineering]  
12+ years in sales | Hackathon Enthusiast | Full Stack & AI Developer  

---

## 📃 License

This project is open-source and free to use under the MIT License.