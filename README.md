Age and Gender Prediction using Deep Learning and OpenCV

This project performs real-time **Age** and **Gender** prediction using your webcam. It uses **OpenCV** for face detection and a pre-trained **Keras** model to predict the age and gender of each detected face.

---
🚀 Features

- Real-time face detection using Haar Cascade.
- Predicts **Age** and **Gender** for each face.
- Displays bounding boxes and predictions on the webcam feed.

---


Install required Python packages:


pip install -r requirements.txt
```

`requirements.txt` content:
```
tensorflow
keras
opencv-python
numpy
```

---

 📁 Project Structure

```
Age_and_gender/
├── model/
│   └── age_gender_model.h5        # Pre-trained model file
├── predict.py                     # Main script
├── README.md                      # Project info
└── tf-env/                        # (Optional) virtualenv folder
```

---
💻 How to Run

1. Clone the repository:
```bash
git clone https://github.com/your-username/AGE_GENDER.git
cd AGE_GENDER
```

2. Place the pre-trained model in the `model/` folder as `age_gender_model.h5`.

3. Run the script:
```bash
python predict.py
```

4. Press `Q` to quit the webcam window.

---

🧠 Model Output Format

The model should output:
- Age as a float in a 2D array: `[[age]]`
- Gender as probabilities: `[[male_prob, female_prob]]`

**Example Output:**
```
Predictions:
[array([[22.13]], dtype=float32), array([[0.92, 0.08]], dtype=float32)]
```

- **Gender:** Use the higher probability (e.g. if male_prob > 0.5, it's Male)
- **Age:** Use `int(predictions[0][0])`

---

⚙️ Optional: Virtual Environment Setup

```bash
python -m venv tf-env
.\tf-env\Scripts\activate   # For Windows
source tf-env/bin/activate # For macOS/Linux
pip install -r requirements.txt
