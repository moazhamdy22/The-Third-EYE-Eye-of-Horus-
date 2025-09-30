# The Third Eye – Intelligent Assistant System for the Blind and Visually Impaired  

## Tested by Testination – ITI Software Testing Team 

<img width="2048" height="1872" alt="2_1" src="https://github.com/user-attachments/assets/abacdd43-e8a5-4299-b875-f483379135f9" />


## 📖 Project Overview  
The **Third Eye** is a smart assistive system designed to empower blind and visually impaired (BVI) individuals with greater independence, safety, and confidence. Unlike traditional aids (e.g., white canes, guide dogs) or single-function apps, this project integrates multiple assistive features into **one cohesive wearable platform**.  

Our solution combines **AI-powered perception, real-time auditory feedback, and ergonomic hardware** to address the four major challenges faced by BVI users:  
- **Navigation & Mobility** (indoor & outdoor obstacle avoidance, GPS guidance)  
- **Environmental Awareness** (object detection & scene understanding)  
- **Social Interaction** (real-time face recognition of familiar people)  
- **Access to Information** (OCR for reading text in English & Arabic)  

An **SOS emergency system** is also embedded, allowing users to send live location alerts via SMS, email, and voice calls at the press of a button.  

---

## ⚙️ Core Functionalities  
1. **Intelligent Navigation Assistant** – Fusion of YOLOv8, ResNet50, Mask2Former, GPS, and ultrasonic sensors for real-time safe path guidance.  
2. **Object Detection & Scene Description** – Identifies and describes objects/people using YOLOv8, BLIP, and Gemini models.  
3. **Face Recognition** – MTCNN + FaceNet pipeline with on-device storage for privacy.  
4. **Optical Character Recognition (OCR)** – Reads printed/digital text in **English and Arabic** using Qwen2.5-VL and Tesseract.  
5. **Emergency SOS System** – One-touch alert via Twilio + Gmail API integration.  
6. **Wearable Hardware** – Lightweight head-mounted sensor unit (camera, ultrasonic sensor, Arduino Nano) + backpack processing unit (Dell Snapdragon laptop).  

---

## 🛠️ System Architecture  
- **Hardware:** Logitech C270 camera, HC-SR04 ultrasonic sensor, Arduino Nano, wireless air mouse, headset, Dell Latitude 7455 processing unit.  
- **Software:** Python, PyTorch, TensorFlow, OpenCV, ultralytics (YOLOv8), Salesforce BLIP, Google Gemini API, Twilio API.  
- **Design Principle:** Modular two-part system (wearable + processing unit) for real-time AI inference and robust performance.  

---

## ✅ Performance Highlights  
- Navigation guidance accuracy: **94.4%**  
- Face recognition precision: **92.1%**  
- OCR accuracy: **95.2%** (English & Arabic)  
- Object detection: **mAP50 = 0.87**  
- SOS reliability: **100% success rate** in testing  

---

## 🌍 Impact & Recognition  
- Funded by **ITIDA – ITAC Graduation Support Program** (Project ID: GP2025.R20.155)  
- Ranked **4th out of 259 teams** in the **10th Dell Technologies Envision the Future 2025 Competition** (Middle East, Africa & Turkey Region)  

---

## 📂 Project Deliverables  
- **Video (Drive):** [https://drive.google.com/file/d/1HfFaEMft03BzChPcF6IbhlY6JQKTtadx/view?usp=drive_link]
- **Technical report (pdf):** [https://drive.google.com/file/d/1SdvVq3mESt6CdYNT59YZJdks8-04ud7f/view?usp=sharing]  

---

## 🚀 Future Directions  
- Hardware miniaturization with Jetson Orin Nano / Qualcomm RB5  
- Advanced **indoor SLAM navigation**  
- Fully hands-free **offline voice command interface**  
- Haptic feedback for silent directional cues  
- Companion mobile app for caregivers  

---

## 👨‍💻 Team & Contributors  
This project was developed by a dedicated team of Biomedical Engineering students at **El Shorouk Academy**,
