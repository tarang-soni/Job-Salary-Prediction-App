# 💼 End-to-End Data Science: Job Salary Prediction (Regression)

## 🎯 Project Overview
This application predicts **job salaries** based on various input features such as role, experience, and other relevant attributes.  
It uses a **Random Forest Regressor** to generate salary estimates through a Streamlit web interface.

---

## 📂 Project Structure
- `data/` → Contains trained model file (`model.pkl`)
- `dataset/` → Raw dataset used for training
- `notebook/` → Jupyter notebook for EDA and model development
- `webapp/` → Streamlit app (`app.py`)
- `requirements.txt` → Project dependencies
- `README.md` → Documentation

---

## 🚀 How to Run the Project

### 1. Clone the Repository
```
git clone https://github.com/tarang-soni/Your-Repository-Name.git
cd Your-Repository-Name
```

### 2. Create & Activate Virtual Environment

**Windows**
```
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux**
```
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```

### 4. Run the Streamlit App
```
streamlit run webapp/app.py
```

### 5. Open in Browser
```
http://localhost:8501
```

---

## 💡 Usage
- Enter job details  
- Click **Predict**  
- View predicted salary  

---

## ⚠️ Notes
- Ensure `model.pkl` exists inside the `data/` folder  
- Activate virtual environment before running  
