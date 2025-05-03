![Screenshot 2025-04-16 110728](https://github.com/user-attachments/assets/9dce4285-e9fb-4991-8ab1-c5d471c82feb)# VOICE AND VISION: Real-Time Two-Way Sign Language Translation

## Overview

**VOICE AND VISION** is a real-time, two-way communication system designed to bridge the gap between hearing-impaired individuals and the general population. It translates **Indian Sign Language (ISL) to text and speech**, and vice versa, using deep learning, computer vision,and 3D animation technologies.
## 🎯 Features

- **Sign-to-Text and Speech Translation**
  - Real-time webcam-based gesture recognition using MediaPipe Holistic
  - LSTM-based model for ISL gesture classification
  - Text and audio output using TTS (gTTS or Web Speech API)

- **Text/Speech-to-Sign Translation**
  - NLP-based grammar translation for ISL sentence structure
  - Blender-rendered 3D animated avatars for sign output
  - Video rendering and retrieval from a MySQL database

- **Full-Stack Web Interface**
  - React.js frontend with animated UI
  - Express.js backend for animation queries
  - MySQL database storing Blender-rendered ISL sign videos
  - 
## 📊 Dataset

The dataset began with samples from the [INCLUDE](https://zenodo.org/records/4010759) ISL dataset. Since it was insufficient for training a deep learning model, we recorded our own videos across **12 ISL action classes**, resulting in **104 videos per class**.

To improve model performance, we augmented the data (e.g., flipping, brightness shifts), expanding it to **520 videos per class**, for a total of **6,240 videos**.

- **Classes**: 12 ISL signs (e.g., hello, thank you, yes, no, etc.)
- **Format**: `.mp4` and `.mov` videos, 2–5 seconds each
- **Keypoints**: Extracted using MediaPipe Holistic and saved in `.npy` format


## 🏗️ System Architecture
                       +-------------------+
                       |  User Interaction |
                       +--------+----------+
                                |
         +----------------------+----------------------+
         |                                             |
 [Sign Language Input]                         [Text / Speech Input]
         |                                             |
 [Webcam Video Capture]                      [Speech-to-Text (STT)]
         |                                             |
 [MediaPipe Holistic Detection]             [ISL Sentence Formatting]
         |                                             |
 [Keypoint Extraction (.npy)]              [Keyword-Based Video Mapping]
         |                                             |
 [Gesture Classification (LSTM)]       [3D Animation Video Retrieval (MySQL)]
         |                                             |
 [Predicted Text Output]               [Blender-Rendered Sign Animation]
         |                                             |
 [Text-to-Speech Conversion]        [Sign Video Display (React Frontend)]
         |                                             |
         +----------------------+----------------------+
                                |
                    +-----------v-----------+
                    |  Real-Time Feedback UI |
                    +------------------------+

This architecture covers:
- **Sign-to-Speech**: Uses webcam input, keypoint extraction, and LSTM classification to generate speech.
- **Speech-to-Sign**: Converts user speech/text to ISL structure and retrieves corresponding animated sign videos.
  
Technologies Used
Computer Vision: MediaPipe Holistic
Deep Learning: LSTM, RNN (TensorFlow/Keras)
Speech: gTTS, Web Speech API
Animation: Blender 3D
Backend: Node.js, Express.js
Database: MySQL
Frontend: React.js
Communication protocal: Websocket,Flask

Sample output:



