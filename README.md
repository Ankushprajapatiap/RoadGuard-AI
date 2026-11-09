RoadGuard AI 🚦
AI-Powered Traffic Accident Prediction and Road Safety Management System

RoadGuard AI is a web-based project designed to support traffic accident analysis, injury severity prediction, accident monitoring, and road safety management.

Features
Dashboard: View accident statistics and key safety indicators.
Accident Records: Add, search, filter, and review accident records.
Accident Verification: Demo workflow for verifying or rejecting accident records.
Accident Map: Visualize accident locations using an interactive map.
AI/ML Prediction: Predict accident injury severity based on input features.
Analytics: Explore accident trends and statistics through charts.
Model Performance: View machine learning evaluation metrics.
Alerts: Monitor road safety alerts.
Theme Support: Light and dark modes.
Backend API: FastAPI endpoints for accident data and dashboard statistics.
Technology Stack
Area	Technologies
Frontend	React, Vite, JavaScript
UI & Charts	Lucide React, Recharts
Maps	Leaflet, React Leaflet, OpenStreetMap
Backend	Python, FastAPI
Database	SQLite
Machine Learning	Scikit-learn, Pandas, Joblib
Project Structure
roadguardAI/
├── backend/
│   ├── main.py
│   ├── ...
│   └── requirements.txt
├── src/
│   ├── App.jsx
│   ├── ...
├── public/
├── index.html
├── package.json
├── .gitignore
└── README.md
The structure above is illustrative; actual files may differ.

Getting Started
1. Clone the repository
git clone https://github.com/YOUR-USERNAME/RoadGuard-AI.git
cd RoadGuard-AI
Replace YOUR-USERNAME with your GitHub username.

2. Run the frontend
npm install
npm run dev
Open the local URL displayed in the terminal, usually:

http://localhost:5173/
3. Run the backend
Open a separate terminal:

cd backend
python -m venv venv
Activate the virtual environment:

Windows PowerShell

.\venv\Scripts\Activate.ps1
Install backend dependencies:

pip install -r requirements.txt
Start FastAPI:

python -m uvicorn main:app --reload
Backend URLs:

API: http://127.0.0.1:8000/
Health check: http://127.0.0.1:8000/health
API documentation: http://127.0.0.1:8000/docs
API Endpoints
Method	Endpoint	Purpose
GET	/	API welcome message
GET	/health	Health check
GET	/accidents	Retrieve accident records
POST	/accidents	Add an accident record
GET	/dashboard/stats	Retrieve dashboard statistics
PUT	/accidents/{accident_id}/verify	Update accident verification
Machine Learning
The current model predicts injury severity categories:

Fatal
Grievous Injury
Minor Injury
No Injury
The previously evaluated model achieved approximately 45.99% accuracy on its test split. Performance should be improved and validated before any real-world or government deployment.

Important: Injury severity prediction is different from predicting whether an accident will occur.

Important Notes
Map markers and sample dashboard figures may be demonstrative and should not be treated as verified live government statistics.
The current officer login is a demonstration implementation, not production-grade authentication.
Do not upload private datasets, passwords, API keys, .env files, or sensitive accident records.
Production deployment requires secure authentication, authorization, data validation, privacy controls, and further model validation.
Future Improvements
Secure backend officer authentication and role-based access.
Automated dataset upload and model retraining.
Improved model performance and evaluation.
Live traffic and weather data integration.
Production deployment and monitoring.
Project Purpose
Developed as an academic and placement portfolio project to demonstrate skills in full-stack web development, REST APIs, data analysis, and machine learning.

RoadGuard AI — Building safer roads through data and AI.