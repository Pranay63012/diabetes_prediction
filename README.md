# 🩺 DIABETES PREDICTION SYSTEM  
### A Machine Learning–powered health risk assessment tool built using Streamlit

🔗 **Live App:** https://diabetesprediction-1.streamlit.app/

---

## 🚀 Project Overview  
The **Diabetes Prediction System** is a user-friendly health analytics tool designed to predict whether an individual is at high risk of diabetes based on medical parameters.  

This project uses a **Random Forest Classifier**, trained on a refined diabetes dataset, along with a modern **Streamlit UI**.

---

## 🧠 Features  
✔️ Interactive, clean, blue–black themed UI  
✔️ Users can enter health details manually  
✔️ Includes **explanations for each input field**  
✔️ Displays a **small sample input sheet** for user reference  
✔️ Real-time prediction using a pre-trained ML model  
✔️ Fully compatible with **Streamlit Cloud deployment**

---

## 📊 Input Features Used  
These features are medically relevant for diabetes risk prediction:

| Feature | Description |
|--------|-------------|
| **Pregnancies** | Number of pregnancies (for female patients) |
| **Glucose Level** | Plasma glucose concentration (mg/dL) |
| **Blood Pressure** | Diastolic blood pressure (mm Hg) |
| **Skin Thickness** | Triceps skin fold thickness (mm) |
| **Insulin** | Serum insulin concentration (µU/mL) |
| **BMI** | Body Mass Index |
| **Diabetes Pedigree Function** | Family history score |
| **Age** | Age of the person |

---


## 🗂️ Project Structure

diabetes_prediction/
│
├── app/
│ └── app.py
│
├── data/
│ └── diabetes.csv
│
├── models/
│ ├── diabetes_rf_model.pkl
│ └── scaler.pkl
│
├── scripts/
│ ├── preprocess.py
│ ├── train.py
│ └── utils.py
│
├── requirements.txt
└── README.md


---

## 🛠️ Technologies Used
- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **Streamlit**
- **Joblib**
- **Matplotlib (for optional graphs)**

---

## 📦 Installation (Local Setup)

### 1️⃣ Clone the repository  

git clone https://github.com/Pranay63012/diabetes_prediction.git
cd diabetes_prediction

pip install -r requirements.txt

streamlit run app/app.py

🌐 Deployment on Streamlit Cloud

Push the project to GitHub

Go to: https://streamlit.io/cloud

Click New App

Select the repo: Pranay63012/diabetes_prediction

Set the app file path:

app/app.py


Deploy 🚀


📘 Sample Input Sheet

A small table is shown inside the app for reference.
Users can manually copy the values into input fields.

🏁 Final Notes

This project is fully optimized for:
✔️ Real-time predictions
✔️ Cloud deployment
✔️ Clean, modern UI
✔️ Beginner-friendly usage



These are the output images 

<img width="1826" height="966" alt="image" src="https://github.com/user-attachments/assets/2d9c4816-3f62-4c9f-af05-1513ebaf4397" />


<img width="1788" height="921" alt="image" src="https://github.com/user-attachments/assets/cf84b507-da68-43f6-a34f-bdf68c82ea42" />



👨‍💻 Author

Pranay Rachakonda
Machine Learning • AI • Data Science Enthusiast



