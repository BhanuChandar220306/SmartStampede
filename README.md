 🚨 SmartStampede
 Intelligent Crowd Safety & Early Warning System

SmartStampede is an AIpowered crowd safety system that transforms CCTV/video surveillance into an intelligent monitoring system.

The system uses a hybrid AI approach combining YOLOv8 for person detection and CSRNet for crowd density estimation. It also analyzes crowd movement and calculates a Crowd Pressure Index (CPI) to identify potentially dangerous crowd conditions.

The monitored area is divided into 8 sectors (A1–A4 and B1–B4) to identify the exact sector where crowd risk is high.



 🎯 Key Features

 🤖 YOLOv8based person detection
 👥 CSRNetbased crowd density estimation
 📊 Crowd Pressure Index (CPI)
 🗺️ 8sector crowd risk localization
 🏃 Crowd movement analysis
 🟢 NORMAL / 🟡 WARNING / 🔴 ALERT status
 🚨 Dangeroussector identification
 🔊 Voice alert support
 📧 Email alert support
 📹 CCTV/video feed processing
 💾 Crowd monitoring history



 🧠 How It Works

             CCTV / Video Feed
                    │
                    ▼
              OpenCV Processing
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
       YOLOv8               CSRNet
   Person Detection    Density Estimation
          │                   │
          └─────────┬─────────┘
                    ▼
             Crowd Analysis
                    │
          ┌─────────┼─────────┐
          │         │         │
          ▼         ▼         ▼
       Density   Movement   Model
                           Disagreement
          │         │         │
          └─────────┼─────────┘
                    ▼
             CPI Calculation
                    │
                    ▼
             Risk Assessment
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
       NORMAL    WARNING     ALERT
                    │
                    ▼
           Dangerous Sector
              Identification
                    │
                    ▼
              Alert System
________________________________________
🛠️ Technologies Used
Technology	Purpose
Python	Main programming language
OpenCV	Video processing
YOLOv8	Person detection
CSRNet	Crowd density estimation
PyTorch	Deep learning
NumPy	Numerical processing
Streamlit	Dashboard/UI
SQLite	Crowd history
gTTS	Texttospeech
pyttsx3	Voice alerts
pygame	Audio playback
Git LFS	CSRNet model storage
________________________________________
💻 System Requirements
Recommended:
•	Windows 10/11
•	Python 3.10 or later
•	Git
•	Git LFS
•	Minimum 8 GB RAM
•	Internet connection for initial package/model setup
A GPU is recommended for better performance, but the system can run on CPU.
________________________________________
🚀 SETUP GUIDE
STEP 1 — Clone the Repository
Open PowerShell or Git Bash:
git clone https://github.com/BhanuChandar220306/SmartStampede.git
Go inside the project:
cd SmartStampede
________________________________________
STEP 2 — Install Git LFS
The project contains the CSRNet trained model:
weights.pth
The file is approximately 65 MB and is stored using Git LFS.
Install/initialize Git LFS:
git lfs install
Download the actual model file:
git lfs pull
Verify that the model exists:
dir weights.pth
The file should be approximately 65 MB.
________________________________________
STEP 3 — Create Python Virtual Environment
Create the environment:
python m venv venv
Activate it:
.\venv\Scripts\activate
You should see:
(venv) PS C:\...\SmartStampede>
________________________________________
STEP 4 — Install Required Packages MANUALLY
Do NOT install requirements.txt.
Install the main packages manually using the commands below.
1. NumPy
pip install numpy
2. OpenCV
pip install opencvpython
3. PyTorch
pip install torch==1.13.1 torchvision==0.14.1
4. Ultralytics YOLO
pip install ultralytics==8.0.145
5. Streamlit
pip install streamlit==1.23.1
6. Pandas
pip install pandas
7. Matplotlib
pip install matplotlib
8. SciPy
pip install scipy
9. Pillow
pip install Pillow
10. Requests
pip install requests
11. PyYAML
pip install PyYAML
12. gTTS
pip install gTTS
13. pyttsx3
pip install pyttsx3
14. pygame
pip install pygame
15. psutil
pip install psutil
________________________________________
⚡ EASY INSTALLATION COMMAND
Instead of running the commands one by one, you can install the required packages using a single command:
pip install numpy opencvpython torch==1.13.1 torchvision==0.14.1 ultralytics==8.0.145 streamlit==1.23.1 pandas matplotlib scipy Pillow requests PyYAML gTTS pyttsx3 pygame psutil
This is the recommended method.
________________________________________
🔍 STEP 5 — Verify the Installation
Run the following commands.
Check Python
python version
Check PyTorch
python c "import torch; print('PyTorch:', torch.__version__)"
Check OpenCV
python c "import cv2; print('OpenCV:', cv2.__version__)"
Check YOLO
python c "from ultralytics import YOLO; print('YOLO: OK')"
Check Streamlit
streamlit version
If these commands work without errors, the environment is ready.
________________________________________
▶️ STEP 6 — RUN SMARTSTAMPEDE
After completing the setup, you only need to run run.py.
python run.py
🚨 IMPORTANT
Do NOT run these files individually:
ai_engine.py
csrnet.py
database.py
email_alert.py
recommendation.py
voice_alert.py
test_ai.py
test_csrnet.py
They are supporting modules.
The main entry point is:
run.py
Simply run:
python run.py
________________________________________
🎥 VIDEO INPUT
The system processes the video/CCTV source configured in:
run.py
If a sample video path is already configured, the system can be started directly.
If you want to use another video, update the video path in run.py.
Example:
VIDEO_PATH = "videos/crowd12.mp4"
Make sure the video exists at the specified location.
________________________________________
📺 SYSTEM OUTPUT
When the system starts, it displays realtime crowd information such as:
YOLO Count
CSRNet Count
Hybrid Count
Crowd Pressure Index
Risk Level
Danger Sector
Example:
STATUS: WARNING

YOLO Count: 42
CSRNet Count: 51

CPI: 63

Danger Sector: B3
During a critical situation:
STATUS: ALERT

Risk Level: HIGH

Danger Sector: B3
________________________________________
🗺️ 8SECTOR MAPPING
The camera frame is divided into:
The system identifies the sector where the highest crowd risk is detected.
┌──────────┬──────────┬──────────┬──────────┐
│    A1    │    A2    │    A3    │    A4    │
├──────────┼──────────┼──────────┼──────────┤
│    B1    │    B2    │    B3    │    B4    │
└──────────┴──────────┴──────────┴──────────┘
________________________________________
📊 CROWD PRESSURE INDEX
The current CPI calculation considers:
60% Density
25% Movement
15% Model Disagreement
The system uses the resulting risk level to generate:
🟢 NORMAL
🟡 WARNING
🔴 ALERT
________________________________________
🔊 ALERT SYSTEM
SmartStampede supports:
Voice Alerts
Voice warnings can be generated when an ALERT condition is detected.
Email Alerts
Email notifications can be sent when email alert functionality is enabled and correctly configured.
Never upload email passwords, API keys, or other credentials to GitHub.
________________________________________
❗ TROUBLESHOOTING
weights.pth not found
Run:
git lfs pull
Then verify:
dir weights.pth
________________________________________
ModuleNotFoundError
Example:
ModuleNotFoundError: No module named 'cv2'
Make sure the virtual environment is active:
.\venv\Scripts\activate
Then install the missing package.
For example:
pip install opencvpython
________________________________________
YOLO model not found
If the YOLO model is not available, make sure the required .pt model is present or allow Ultralytics to download it when the system first runs.
________________________________________
Video cannot be opened
Check the video path in:
run.py
Make sure the video exists and is accessible.
________________________________________
Program is slow
AIbased crowd analysis requires significant processing power.
For better performance:
•	Use a GPU if available.
•	Use a lowerresolution video.
•	Close unnecessary applications.
•	Use a smaller input video.
________________________________________
🔐 SECURITY
Never upload the following to GitHub:
Passwords
API Keys
Email Credentials
Authentication Tokens
Private Keys
The CSRNet model weights.pth is already stored in Git LFS.
________________________________________
🎯 APPLICATIONS
SmartStampede can be used for:
🏟️ Stadiums
🚉 Railway Stations
🎪 Festivals
🛕 Religious Gatherings
🎤 Concerts
🏫 College Events
🏙️ Public Gatherings
🚇 Transportation Hubs
________________________________________
🚀 QUICK START
For a new laptop:
git clone https://github.com/BhanuChandar220306/SmartStampede.git

cd SmartStampede

git lfs install

git lfs pull

python m venv venv

.\venv\Scripts\activate

pip install numpy opencvpython torch==1.13.1 torchvision==0.14.1 ultralytics==8.0.145 streamlit==1.23.1 pandas matplotlib scipy Pillow requests PyYAML gTTS pyttsx3 pygame psutil

python run.py
⭐ After setup
Every time you want to run the project:
cd SmartStampede
.\venv\Scripts\activate
python run.py
________________________________________
👥 TEAM
Team ID: 18
Project: SmartStampede
Problem Statement:
Transforming CCTV Surveillance into an Intelligent Crowd Safety System
________________________________________
🏆 PROJECT GOAL
SmartStampede aims to transform passive CCTV surveillance into intelligent and proactive crowd safety monitoring by detecting risky crowd conditions early, identifying the affected sector, and supporting faster intervention.
🚨 Detect Early. Identify Danger. Respond Faster.

 


