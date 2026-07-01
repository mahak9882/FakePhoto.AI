# 📸 FakePhoto.AI – Spot the Fake Photo

A lightweight Computer Vision and Machine Learning system for detecting whether an image is a **real photograph** or a **photo of a screen (screen recapture)**.

This project was developed as part of a Computer Vision & Machine Learning take-home assignment. The objective was to build a fast, explainable, and CPU-friendly solution capable of distinguishing between genuine camera photographs and images captured from digital screens.

---
<img width="1910" height="907" alt="image" src="https://github.com/user-attachments/assets/30fac1a9-c2b9-490f-81b8-fdcb81940878" />


## 🚀 Features

* Detects **Real Photos** vs **Screen Recaptures**
* Classical Computer Vision + Machine Learning approach
* No deep learning required
* Runs completely offline
* Lightweight and CPU-friendly
* Fast inference
* Probability-based prediction (0 = Real, 1 = Screen)

---

## 🛠️ Tech Stack

* Python
* OpenCV
* NumPy
* SciPy
* Scikit-learn
* Scikit-image
* XGBoost
* Joblib
* Matplotlib

---

## 📂 Project Structure

```text
FakePhoto.AI/
│
├── dataset/
│   ├── real/
│   └── screen/
│
├── feature_extraction.py
├── train.py
├── predict.py
├── model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
│
├── confusion_matrix.png
├── roc_curve.png
├── feature_importance.png
└── predictions.csv
```

---

## ⚙️ Approach

Instead of training a deep learning model, this project uses handcrafted Computer Vision features combined with Machine Learning.

### Workflow

```
Input Image
      │
      ▼
Pre-processing
      │
      ▼
Feature Extraction
      │
      ▼
Feature Scaling
      │
      ▼
Machine Learning Model
      │
      ▼
Probability Score
```

---

## 🔍 Feature Extraction

The following image features are extracted for every image:

* Fast Fourier Transform (FFT)
* Local Binary Patterns (LBP)
* Gray Level Co-occurrence Matrix (GLCM)
* Histogram of Oriented Gradients (HOG)
* Laplacian Variance (Sharpness)
* Edge Density (Canny)
* RGB Colour Statistics
* Reflection Detection
* Grid Regularity Features

These handcrafted features capture texture, frequency, colour, and structural patterns that help differentiate real-world photographs from screen recaptures.

---

## 🤖 Machine Learning Models

The following models were explored:

* Logistic Regression
* Random Forest
* XGBoost

Feature scaling was performed using **StandardScaler**, and hyperparameters for XGBoost were optimized using **GridSearchCV**.

---

## 📊 Model Evaluation

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score
* Confusion Matrix
* 5-Fold Cross Validation

Example evaluation output:

```
Accuracy : 78.38%

Precision : 0.79

Recall : 0.78

F1 Score : 0.78

ROC AUC : 0.844
```
<img width="875" height="645" alt="image" src="https://github.com/user-attachments/assets/07afd3c4-201d-4f9d-8c2e-0a711527152c" />

> Replace these values with your latest results if you further improve the model.

---

## 📈 Output Visualizations

The training script automatically generates:

* Confusion Matrix
* ROC Curve
* Feature Importance Graph
* Prediction CSV

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/mahak9882/FakePhoto.AI.git

cd FakePhoto.AI
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📁 Dataset Structure

```
dataset/

├── real/
│     image1.jpg
│     image2.jpg
│
└── screen/
      image1.jpg
      image2.jpg
```

---
<img width="706" height="317" alt="image" src="https://github.com/user-attachments/assets/65be6675-2ab4-4046-b198-9d15865162d6" />

## 🏋️ Training

Run:

```bash
python train.py
```

This will:

* Extract image features
* Train the model
* Perform cross-validation
* Evaluate performance
* Save:

  * `model.pkl`
  * `scaler.pkl`

---

## 🔍 Prediction

Predict a single image:

```bash
python predict.py sample.jpg
```

Example output:

```
0.91

Prediction : SCREEN PHOTO
Confidence : 0.91
```

or

```
0.08

Prediction : REAL PHOTO
Confidence : 0.08
```

---

## 📌 Future Improvements

* Collect a larger and more diverse dataset
* Improve feature engineering for difficult cases
* Better handling of printouts and high-resolution displays
* Explore lightweight deep learning models such as MobileNetV3
* Deploy the model on Android/iOS using OpenCV Mobile SDK and ONNX Runtime

---

## 📚 Learning Outcomes

This project helped strengthen my understanding of:

* Computer Vision
* Image Processing
* Feature Engineering
* Machine Learning
* Model Evaluation
* Hyperparameter Tuning
* OpenCV
* XGBoost
* Python Development

---

## 👩‍💻 Author

**Mahak Taneja**

B.Tech Computer Science Engineering

GitHub: https://github.com/mahak9882

---

## 📄 License

This project is licensed under the MIT License.
