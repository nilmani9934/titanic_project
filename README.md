Titanic Survival Prediction – Machine Learning Project

This project builds a Machine Learning model to predict whether a passenger survived the Titanic disaster based on features like age, gender, ticket class, family size, fare, and more.

The project includes:

Data preprocessing

Exploratory Data Analysis (EDA)

Feature engineering

Model training & comparison

Final model deployment (Random Forest)

📂 Project Files
titanic_project/
│
├── titanic_project.py        # Full ML implementation
├── titanic.csv               # Dataset used in this project
├── requirements.txt          # Python dependencies for Binder
└── runtime.txt               # Python runtime version

🎯 Project Objectives

Understand Titanic dataset and extract insights

Perform data cleaning & preprocessing

Build predictive ML models

Compare accuracy of multiple algorithms

Select best model (Random Forest)

Visualize relationships between features & survival

📊 Machine Learning Models Used

The following models were trained and evaluated:

Model	Accuracy
Logistic Regression	~80%
Random Forest	~84%
SVM	~82%
KNN	~76%

➡️ Random Forest was selected as the best-performing model.

🔍 Key Insights from Data

Females survived significantly more than males

1st class passengers had the highest survival rate

Children had higher chances of survival

Family size and titles derived from names improved model accuracy

📦 Installation Instructions
1. Clone the repository
git clone https://github.com/<your-username>/titanic_project.git
cd titanic_project

2. Install dependencies
pip install -r requirements.txt

3. Run the project
python titanic_project.py

🚀 Run the Project in the Cloud (Binder)

After uploading your files to GitHub, use this link:

👉 https://mybinder.org/v2/gh/nilmani9934/titanic_project/main?lab=classic

This will open your project in a fully hosted Jupyter Notebook environment.

🧠 Technologies Used

Python

Pandas, NumPy

Seaborn, Matplotlib

Scikit-Learn

Joblib

Binder (for notebook hosting)

📈 Future Improvements

Hyperparameter tuning

Add interactive Streamlit Web App

Add neural network model

Deploy as API using Flask / FastAPI

👤 Author

Nilmani Singh
Machine Learning & Data Analytics Enthusiast
