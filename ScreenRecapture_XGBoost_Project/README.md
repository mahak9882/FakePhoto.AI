# Screen Recapture Detection

This starter project uses handcrafted computer vision features
(FFT, LBP, GLCM, HOG, color, edges, Laplacian, reflections)
with XGBoost for classifying:
0 = Real photo
1 = Photo of a screen

Files to implement:
- feature_extraction.py
- train.py
- predict.py

Install:
pip install opencv-python scikit-image scikit-learn xgboost joblib numpy
