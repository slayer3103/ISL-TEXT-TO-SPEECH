# 🤟 ISL Text-to-Speech Translator

A real-time **Indian Sign Language (ISL) to Text and Speech Translator** that uses **computer vision + machine learning** to convert hand gestures into readable text and audible speech.

This project focuses on **accessibility**, enabling smoother communication for speech and hearing-impaired individuals through a lightweight, real-time system.

---

## 🚀 Highlights

- Real-time gesture recognition using webcam
- Landmark-based approach (efficient & fast)
- ML-based gesture classification
- Text + Speech output pipeline
- Modular and scalable project structure

---

## 🧠 Tech Stack

- **Computer Vision:** OpenCV, MediaPipe
- **Machine Learning:** Scikit-learn
- **Backend:** Python
- **Frontend:** HTML, CSS, JavaScript
- **Audio:** Text-to-Speech (TTS)

---

## 📂 Project Structure

```
ISL-TEXT-TO-SPEECH/
│── data/
│   ├── recorded_keypoints.csv
│   ├── recorded_keypoints_test.csv
│
│── models/
│   └── trained_model.pkl
│
│── src/
│   ├── data_collection.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── inference.py
│
│── frontend/
│── backend/
│── utils/
│
│── main.py
│── requirements.txt
│── README.md
```

---

## ⚙️ How It Works

1. **Capture** → Webcam captures hand gestures
2. **Extract** → MediaPipe detects 21 hand landmarks
3. **Transform** → Landmarks converted into feature vectors
4. **Predict** → ML model classifies gesture
5. **Output** → Text displayed + converted to speech

---

## 📊 Dataset

Custom dataset built using MediaPipe hand landmarks:

- **recorded_keypoints.csv**
  - Training dataset
  - Label + ~84 landmark features

- **recorded_keypoints_test.csv**
  - Testing dataset
  - Includes landmark + metadata

---

## 🛠️ Setup & Installation

```bash
git clone https://github.com/your-username/ISL-TEXT-TO-SPEECH.git
cd ISL-TEXT-TO-SPEECH
pip install -r requirements.txt
python main.py
```

---

## 🎯 Use Cases

- Assistive tech for speech/hearing impaired
- Gesture-based interfaces
- Real-time ML applications
- Human-computer interaction systems

---

## 📈 Future Work

- Add more gesture classes
- Improve accuracy with deep learning
- Sentence formation (NLP)
- Web/mobile deployment

---

## 🤝 Contribution

Pull requests are welcome. For major changes, open an issue first.

---

## 📜 License

MIT License

---

## 👨‍💻 Author

Developed as part of an accessibility-focused ML project.

---

⭐ Star this repo if you found it useful!
