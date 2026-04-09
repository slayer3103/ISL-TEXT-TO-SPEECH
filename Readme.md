# 🤟 ISL Text-to-Speech Translator

A real-time **Indian Sign Language (ISL) to Text and Speech Translator** that leverages **computer vision and machine learning** to convert hand gestures into meaningful text and audio output.

This project is designed as an **accessibility-focused solution**, enabling communication for speech and hearing-impaired individuals using a lightweight, real-time pipeline.

---

## 🚀 Key Features

- 🎥 Real-time hand gesture recognition via webcam
- 🧠 Landmark-based feature extraction using MediaPipe
- 🤖 Machine Learning model for gesture classification
- 🔊 Text-to-Speech output
- ⚡ Optimized for low-latency inference

---

## 🧠 Tech Stack

- **Computer Vision:** OpenCV, MediaPipe
- **Machine Learning:** Scikit-learn
- **Backend:** Python
- **Frontend:** HTML, CSS, JavaScript
- **Audio Processing:** pyttsx3

---

## 📂 Project Structure

```
ISL-TEXT-TO-SPEECH/
│── backend/
│   ├── app.py
│   ├── routes/
│   └── controllers/
│
│── frontend/
│   ├── index.html
│   ├── styles/
│   └── scripts/
│
│── data/
│   ├── recorded_keypoints.csv
│   ├── recorded_keypoints_test.csv
│
│── models/
│   ├── trained_model.pkl
│
│── utils/
│   ├── mediapipe_utils.py
│   ├── feature_extraction.py
│
│── main.py
│── requirements.txt
│── README.md
```

---

## ⚙️ System Workflow

1. **Input Capture**
   - Webcam captures real-time hand gestures

2. **Landmark Detection**
   - MediaPipe extracts 21 hand keypoints

3. **Feature Engineering**
   - Landmarks are flattened into feature vectors

4. **Model Prediction**
   - Trained ML model classifies the gesture

5. **Output Generation**
   - Gesture is displayed as text
   - Converted into speech using TTS

---

## 📊 Dataset

Custom dataset created using MediaPipe hand landmark extraction:

### 🔹 recorded_keypoints.csv

- Training dataset
- Structure:
  - Label (gesture class)
  - ~84 numerical landmark features

### 🔹 recorded_keypoints_test.csv

- Testing / validation dataset
- Includes additional metadata for debugging and evaluation

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ISL-TEXT-TO-SPEECH.git
cd ISL-TEXT-TO-SPEECH
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python main.py
```

---

## 🎯 Use Cases

- Assistive communication for speech/hearing impaired individuals
- Gesture recognition systems
- Real-time AI-based applications
- Human-computer interaction

---

## 📈 Future Enhancements

- Expand gesture vocabulary
- Integrate deep learning models (LSTM/Transformers)
- Sentence-level translation using NLP
- Deploy as a web/mobile application
- Multi-hand gesture recognition

---

## 🤝 Contribution

Contributions are welcome!

1. Fork the repository
2. Create a new branch
3. Submit a pull request

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

Developed as part of a project focused on **AI-driven accessibility and real-time gesture recognition systems**.

---

⭐ If you found this project useful, consider giving it a star!
