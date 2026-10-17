# 🧠 Parkinson Detection using Deep Learning

This project predicts Parkinson’s disease from **hand‑drawn Spiral and Wave tests** using deep learning models (EfficientNetB0).  
It provides a backend API (Flask) and a frontend webpage (HTML/JS) for interactive predictions.

---

## 📌 Problem Statement
Parkinson’s disease is a progressive neurological disorder that affects movement.  
Doctors often use **Spiral and Wave drawings** to detect motor impairments.  
This project leverages machine learning to classify drawings as **Healthy** or **Parkinson**.

---

## 📊 Dataset
- **Spiral drawings**: Healthy vs Parkinson samples  
- **Wave drawings**: Healthy vs Parkinson samples  
- Images resized to `224x224` and normalized  
- Augmentation applied (flip, rotation, zoom) to improve generalization

---

## ⚙️ Methodology
1. **Data Preprocessing**  
   - Normalization (`Rescaling(1./255)`)  
   - Augmentation (flip, rotation, zoom)  

2. **Model Architecture**  
   - Base: `EfficientNetB0` (pretrained on ImageNet)  
   - Fine‑tuned last 20 layers  
   - Global Average Pooling + Dropout + Dense (sigmoid)  

3. **Training**  
   - Optimizer: Adam (`lr=1e-5`)  
   - Loss: Binary Crossentropy  
   - Early stopping to prevent overfitting  

4. **Evaluation**  
   - Accuracy, Precision, Recall, F1‑score  
   - Confusion matrix for class distribution  

---

## 📈 Results
| Model   | Dataset | Test Accuracy |
|---------|---------|---------------|
| Spiral  | Spiral drawings | ~85% |
| Wave    | Wave drawings   | ~83% |

*(Exact values depend on dataset split and training run)*

---

## 🚀 How to Run Locally
### Backend (Flask)
```bash
git clone https://github.com/TSSHARAN-2007/Parkinson_Detection.git
cd Parkinson_Detection/backend

pip install -r requirements.txt
python app.py
