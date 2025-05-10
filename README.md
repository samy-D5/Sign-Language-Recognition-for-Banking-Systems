# Hand Gesture Recognition & Arduino Communication System

This project enables real-time hand gesture recognition using computer vision and machine learning, then sends the recognized gestures to an Arduino board to perform various hardware actions.

## 📁 Project Structure

- `collect_imgs.py` — Collects image data for each hand gesture using webcam.
- `create_dataset.py` — Creates a dataset from the collected images for model training.
- `train_classifier.py` — Trains a CNN model using Keras for gesture classification.
- `inference_classifier.py` — Performs real-time gesture classification using webcam.
- `sent_to_arduino.py` — Sends recognized characters to an Arduino over serial communication.
- `arduino_communication.py` — Handles Arduino communication (for test/demo).

## 🚀 How to Run

### 1. Install Requirements

```bash
pip install -r requirements.txt
