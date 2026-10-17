# 🧠 Parkinson’s Disease Detection (Final Version)

This repository contains the final notebook [`Parkinson_Disease_Detection_Final.ipynb`](./Parkinson_Disease_Detection_Final.ipynb), which implements a deep learning pipeline to detect **Parkinson’s disease** from **Spiral and Wave drawings**.  
The project combines preprocessing, transfer learning (EfficientNetB0), evaluation, and model saving for deployment.

---

## 📌 Problem Statement
Parkinson’s disease is a progressive neurological disorder that affects motor control.  
Doctors often use **Spiral and Wave drawing tests** to identify motor impairments.  
This project leverages machine learning to classify drawings as **Healthy** or **Parkinson**, providing a potential tool for early screening.

---

## 📊 Dataset
- **Spiral drawings**: Healthy vs Parkinson samples  
- **Wave drawings**: Healthy vs Parkinson samples  
- Preprocessing: resize to `224x224`, normalize pixel values  
- Augmentation: random flips, rotations, zooms  

---

## ⚙️ Methodology
1. **Data Preprocessing**  
   - Normalization (`Rescaling(1./255)`)  
   - Augmentation for robustness  

2. **Model Architecture**  
   - Base: `EfficientNetB0` (pretrained on ImageNet)  
   - Fine‑tuned last 30 layers  
   - Global Average Pooling + Dropout + Dense (sigmoid)  

3. **Training**  
   - Optimizer: Adam (`lr=1e-5`)  
   - Loss: Binary Crossentropy  
   - Early stopping for generalization  

4. **Evaluation**  
   - Accuracy, Precision, Recall, F1‑score  
   - Confusion matrix for class distribution  

---

## 📈 Results
| Model   | Dataset | Test Accuracy |
|---------|---------|---------------|
| Spiral  | Spiral drawings | ~76.7% |
| Wave    | Wave drawings   | ~90% |

*(Exact values depend on dataset split and training run)*

---

## 🚀 How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/TSSHARAN-2007/Parkinson_Detection.git
   cd Parkinson_Detection
