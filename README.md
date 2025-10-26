
````markdown
# Age and Gender Prediction using Deep Learning and OpenCV

This project performs real-time Age and Gender prediction using your webcam. It uses OpenCV for face detection and a pre-trained Keras model to predict the age and gender of each detected face.



## 🚀 Features

- Real-time face detection using Haar Cascade.
- Predicts Age and Gender for each detected face.
- Displays bounding boxes and predictions on the webcam feed.



## 📦 Requirements

Install required Python packages:

```bash
pip install -r requirements.txt
````

`requirements.txt` content:

```
tensorflow
keras
opencv-python
numpy
```

---

## 📁 Project Structure

```
Age_and_gender/
├── model/
│   └── age_gender_model.h5        # Pre-trained Keras model
├── predict.py                     # Main script
├── README.md                      # Project info
└── tf-env/                        # (Optional) virtual environment folder
```

---

## 💻 How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/AGE_GENDER.git
cd AGE_GENDER
```

2. Place the pre-trained model in the `model/` folder as `age_gender_model.h5`.

3. Run the main script:

```bash
python predict.py
```

4. Press **Q** to quit the webcam window.

---

## 🧠 Model Output Format

The model outputs:

* **Age** as a float in a 2D array: `[[age]]`
* **Gender** as probabilities: `[[male_prob, female_prob]]`

**Example Output:**

```python
Predictions:
[array([[22.13]], dtype=float32), array([[0.92, 0.08]], dtype=float32)]
```

* **Gender:** Use the higher probability (e.g., if `male_prob > 0.5`, it's Male)
* **Age:** Use `int(predictions[0][0])`

---

## ⚙️ Optional: Virtual Environment Setup

```bash
# Create virtual environment
python -m venv tf-env

# Activate virtual environment
# Windows
.\tf-env\Scripts\activate
# macOS/Linux
source tf-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📌 Notes

* Make sure your webcam is connected and accessible.
* Ensure that the pre-trained model file (`age_gender_model.h5`) is in the correct folder.
* This project is intended for educational and demonstration purposes.

```
