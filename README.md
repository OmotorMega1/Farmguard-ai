🌿 FarmGuard AI

Crop disease detection and AI-powered treatment recommendations for Nigerian farmers.

A farmer uploads a leaf photo → FarmGuard-AI detects the disease → Generates treatment advice → Results displayed on a Streamlit dashboard.

📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [🛠️ How This Project Was Built](#️-how-this-project-was-built-step-by-step-journey)
- [Getting Started](#getting-started)
- [Running the App](#running-the-app)
- [API Endpoints](#api-endpoints)
- [Environment Variables](#environment-variables)
- [Running Tests](#running-tests)
- [Docker](#docker)
- [Acknowledgements](#acknowledgements)

Overview

FarmGuard AI helps Nigerian farmers identify crop diseases early and receive instant treatment recommendations — without needing an agricultural expert on-site.

Problem: Farmers lose significant crop yields because they cannot identify diseases early enough.

Solution: Upload a leaf photo → get an instant AI-powered diagnosis and treatment plan.

## Features

- **Disease Detection** — CNN model (MobileNetV2 Transfer Learning) detects 6 crop disease classes with 97.3% accuracy
- **Severity Prediction** — Confidence-based severity mapping (Mild / Moderate / Severe)
- **AI Treatment Recommendations** — Groq AI (Llama 3.3) generates contextual treatment advice per disease
- **Healthy Crop Detection** — Generates encouragement messages for healthy crops
- **REST API** — FastAPI backend with 4 endpoints
- **Interactive Dashboard** — Streamlit frontend with dark theme and real-time results

## Tech Stack
Layers and Technology.

```
Model: TensorFlow 2.18, Keras, MobileNetV2 
Backend: FastAPI, Uvicorn 
Frontend: Streamlit 
Gen AI: Groq API (Llama 3.3-70b-versatile) 
Image Processing: Pillow, NumPy 
ML Utilities: Scikit-learn 
Containerization: Docker, Docker Compose 
Environment: Python 3.10, python-dotenv 

```
## Project Structure

```
farmguard-ai/
├── api/
│   └── main.py                 # FastAPI backend — 4 endpoints
├── dashboard/
│   └── app.py                  # Streamlit farmer-facing dashboard
├── data/
│   ├── raw/PlantVillage/       # Original dataset
│   └── split/                  # Train / val / test split
│       ├── train/
│       ├── val/
│       └── test/
├── models/
│   └── disease_model.h5        # Trained CNN model
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_cnn_training.ipynb
├── src/
│   ├── predict.py              # Inference pipeline (CNN + Groq AI)
│   └── train_cnn.py            # Model training script
├── tests/
│   └── test_predict.py         # Test suite (5 tests)
├── .env                        # API keys (not committed)
├── .dockerignore
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Model Performance

| Metric | Value |
 
| Architecture | MobileNetV2 + Custom Dense Layers |
| Training technique | Transfer Learning (ImageNet) |
| Regularization | L2 (λ=0.01) + Data Augmentation |
| Best val accuracy | 98.3% |
| Test accuracy | **97.32%** |
| Total classes | 6 |
| Test set size | 934 images |


### Classification Report:
                               precision    recall  f1-score   

Pepper__bell___Bacterial_spot       0.93      0.92      0.92       
       Pepper__bell___healthy       0.94      0.98      0.96       
        Potato___Early_blight       1.00      0.98      0.99       
             Potato___healthy       1.00      0.96      0.98        
          Tomato_Early_blight       0.99      0.95      0.97       
               Tomato_healthy       0.99      1.00      0.99       

### Overall:                        0.97      0.97      0.97


## 🛠️ How This Project Was Built (Step-by-Step Journey)

# This section walks you through every decision and milestone taken to build FarmGuard AI from scratch. Read this before touching any code — understanding the reasoning behind each step will make the entire codebase make sense.

### Step 1 — Define the Problem

Before writing a single line of code, we defined exactly what we were building and why.

The question: What is the real problem Nigerian farmers face?

The answer: Crop diseases spread undetected because farmers have no fast, affordable way to identify them. By the time a disease is visible enough to recognise manually, it has already caused significant yield loss.

The decision: Build a lightweight AI assistant that any farmer with a smartphone can use — no agricultural expertise required.

### Step 2 — Set Up the Development Environment

Every professional ML project starts with a clean, isolated environment.

```
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install all dependencies
pip install -r requirements.txt
```

# Why a virtual environment?
Without it, every project on your laptop shares the same packages. When two projects need different versions of the same package, things break. A virtual environment gives each project its own isolated space.

We also created the full folder structure (`api/`, `src/`, `dashboard/`, `models/`, `tests/`) before writing any code. This is intentional — having a clear structure from day one prevents messy, disorganised codebases later.

### Step 3 — Explore the Data (`01_data_exploration.ipynb`)

We downloaded the **PlantVillage dataset** from Kaggle — 6,218 leaf images across 6 disease classes. Before training anything, we explored the data thoroughly in a Jupyter notebook.

**Why Jupyter for exploration?**
Jupyter notebooks are ideal for data exploration because you can run code cell by cell, see outputs immediately, and visualise data inline. They are for experimentation — not production code.

**What we discovered:**
```
| Class | Images |:
| Tomato Healthy | 1,591 |
| Pepper Healthy | 1,478 |
| Tomato Early Blight | 1,000 |
| Potato Early Blight | 1,000 |
| Pepper Bacterial Spot | 997 |
| Potato Healthy | 152 ← problem |

# Key finding: Potato Healthy had only 152 images compared to 1,000+ for other classes. This is called **class imbalance**. If left unaddressed, the model would barely learn to recognise Potato Healthy — it simply would not see it enough during training to learn its patterns.

# Data split — 70% / 15% / 15%:**
```

```
Training set (70%)   → 4,350 images  — model learns from this
Validation set (15%) →   934 images  — monitors progress during training
Test set (15%)       →   934 images  — final honest evaluation, touched only once
```

The test set is kept completely separate and only evaluated once at the very end. Using it during training would give an inflated accuracy that does not reflect real-world performance.

### Step 4 — Build and Train the CNN Model (`02_cnn_training.ipynb`)

This is the core of the project. We built the disease detection model step by step inside a Jupyter notebook before moving the clean final code to `src/train_cnn.py`.

#### Why Transfer Learning?

Training a CNN from scratch requires millions of images and weeks of compute time. Instead, we used **Transfer Learning** with **MobileNetV2** — a model Google trained on 1.2 million images (ImageNet).

```
MobileNetV2 already knows how to detect:
  Layer 1–30   → edges and corners
  Layer 31–80  → textures and patterns
  Layer 81–154 → complex shapes and objects

We borrowed all 154 layers (frozen) and added
our own 3 custom layers on top to learn crop diseases.
Only our 3 layers were trained — saving time and compute.
```

# Why MobileNetV2 specifically? 
The "Mobile" in the name means it was designed to be lightweight and fast — perfect for a system that needs to run on basic smartphones used by Nigerian farmers.

#### Model Architecture

```
Input (224 × 224 × 3 leaf image)
        ↓
MobileNetV2 — 154 layers FROZEN
(pretrained on ImageNet — not retrained)
        ↓
GlobalAveragePooling2D
(flattens 3D feature map into a 1D vector)
        ↓
Dense (128 neurons, ReLU activation, L2 regularisation λ=0.01)
(learns disease-specific visual patterns)
        ↓
Dense (6 neurons, Softmax activation)
(outputs a probability for each of the 6 disease classes)
```

# Why 128 neurons in the hidden layer?
Too few neurons = model cannot learn complex patterns (underfitting). Too many neurons = model memorises training data (overfitting). 128 is the right balance for our dataset of 4,350 training images.

# Why Softmax on the output layer?
Softmax converts 6 raw scores into probabilities that sum to exactly 1. The class with the highest probability becomes the prediction.

```
Example output:
[0.02, 0.01, 0.02, 0.01, 0.91, 0.03]
                          ↑
             Tomato Early Blight — 91% confident
```

### Solving the Class Imbalance

We addressed the Potato Healthy imbalance using class weights. — telling the model that mistakes on rare classes are more costly during training.

```python
class_weights = compute_class_weight(class_weight='balanced', ...)

# Resulting weights:
# Potato Healthy:        6.84  ← every mistake costs 6.84× more
# Tomato Healthy:        0.65  ← common class, lower penalty
```

Every time the model got a Potato Healthy image wrong, it paid 6.84× more in loss — forcing it to pay extra attention to that class despite having only 152 images.

### Preventing Overfitting

We used three techniques to stop the model from memorising training images instead of learning general patterns:
```

1. L2 Regularisation (λ=0.01): Adds a penalty for large weights — forces distributed learning across all neurons 
2. Data Augmentation: Randomly rotates, flips, zooms training images — model never sees the exact same image twice 
3. EarlyStopping (patience=5): Automatically stops training when validation loss stops improving for 5 consecutive epochs 

```

### Training Configuration

```
CONFIG = {
    "image_size": (224, 224),  # MobileNetV2 standard input size
    "batch_size": 32,          # images per gradient descent step (standard default)
    "epochs": 20,              # max training rounds
    "learning_rate": 0.001,    # Adam optimizer starting rate
    "num_classes": 6           # one output neuron per disease class
}
```

# Why batch size 32? 
It is the industry standard. Small enough to fit in memory, large enough to give stable gradient estimates. Powers of 2 (16, 32, 64) align with computer memory architecture for efficiency.

# Why image size 224×224?
This is the exact input size MobileNetV2 was trained on. We must match it for Transfer Learning to work.

### Results

```
Best validation accuracy:   98.3%  (epoch 18)
Final test accuracy:        97.32% (934 completely unseen images)

Bias/Variance diagnosis:
  J_train = ~3.8%  → low ✅ (not underfitting)
  J_cv    = ~1.7%  → low ✅ (not overfitting)
  Gap     = ~2.1%  → tiny ✅ (generalises well)
  Verdict: Just right
```

The trained model was saved to `models/disease_model.h5`.

Why save as?  `.h5`?
This file contains all the learned weights and the model architecture. Loading it takes seconds — no retraining required.


### Step 5 — Build the Inference Pipeline (`src/predict.py`)

The trained model sitting in a `.h5` file does nothing on its own. We needed code to load it, process new images, make predictions and generate recommendations. That code lives in `src/predict.py`.

**Why move from Jupyter to a `.py` file?**
Jupyter notebooks are for experimentation. Production code belongs in `.py` files — they are cleaner, importable by other modules, and version-controllable.

# Key functions:
```
| Function | Purpose |
 `preprocess_image()`: Resizes farmer's photo to 224×224, normalises pixels to 0–1 
 `predict_disease()`: Runs image through CNN, returns disease + confidence + severity 
 `get_severity()`: Maps confidence score to Mild / Moderate / Severe 
 `generate_treatment()`: Calls Groq AI to write a contextual treatment recommendation 
 `analyze_leaf()`: Master function — coordinates all of the above in sequence 
```
**The complete flow inside `predict.py`:**

```
Farmer's photo (any size, any format)
        ↓  preprocess_image()
Resized to 224×224, pixels normalized to 0–1
        ↓  predict_disease()
CNN returns: disease="Tomato Early Blight", confidence=0.98, severity="Severe"
        ↓  generate_treatment()
Groq AI generates: "Tomato Early Blight is a fungal disease..."
        ↓  analyze_leaf()
Returns complete result as a Python dictionary
```

# Why Groq AI instead of hardcoded text?

Hardcoded text always returns the same response regardless of context. Groq AI generates advice based on the specific disease, confidence level and severity — a mild infection gets different advice from a severe one, which is far more useful for a farmer.


### Step 6 — Build the REST API (`api/main.py`)

The inference pipeline needed to be accessible over a network so any interface — a web dashboard, a mobile app, a WhatsApp bot — could call it. We used **FastAPI** to build a REST API with 4 endpoints.

# Why FastAPI?
- Fast to build and run
- Automatically generates interactive API documentation at `/docs`
- Modern Python, clean syntax

# The 4 endpoints:

```
GET  /          → confirms the server is running
GET  /health    → checks model and API are both operational
POST /predict   → receives a leaf image, returns full disease analysis
GET  /classes   → lists all 6 supported disease classes
```

**How the `/predict` endpoint works:**

```
1. Receives leaf image uploaded from dashboard
2. Validates it is actually an image file (not a PDF, Word doc, etc.)
3. Saves it temporarily with a UUID filename (prevents filename collisions)
4. Calls analyze_leaf() from predict.py
5. Returns full JSON result to the dashboard
6. Deletes the temporary file — always, even if prediction fails
```


### Step 7 — Build the Dashboard (`dashboard/app.py`)

With a working API, we built the farmer-facing interface using **Streamlit** — a Python library that creates interactive web apps without HTML, CSS or JavaScript.

# Why Streamlit?
- Build web apps entirely in Python
- Perfect for ML demos and production tools
- Rapid development — a full dashboard in one file

# How the dashboard communicates with the API:

```
# When farmer clicks "Analyze Leaf":
uploaded_file.seek(0)  # reset file cursor
files = {"file": (name, file, content_type)}
response = requests.post("http://127.0.0.1:8000/predict", files=files)
result = response.json()

# Display results:
st.write(result["disease"])
st.write(result["confidence"])
st.write(result["recommendation"])
```

# Colour-coded results by severity:

```
🟢 Healthy   → green success box + balloons animation
🔴 Severe    → red error box + urgent warning banner
🟠 Moderate  → orange warning box
🔵 Mild      → blue info box
```

### Step 8 — Write Automated Tests (`tests/test_predict.py`)

Before the system reaches real farmers, we need confidence that every part works correctly. We wrote 5 automated tests that can be run in seconds.

```
python tests/test_predict.py
```

# Test coverage:

```
Test 1 → Model file loads from disk without errors
Test 2 → Prediction returns all required fields with valid values
Test 3 → Model correctly identifies images from all 6 disease classes
Test 4 → Healthy crops are correctly flagged as is_healthy=True
Test 5 → Severity levels map correctly to confidence thresholds
```

The test file imports directly from `predict.py` — no duplicate code. If `predict.py` changes, the tests automatically reflect those changes.


### Step 9 — Containerise with Docker

Finally, we packaged the entire application into Docker containers so it runs consistently on any machine — no manual Python setup, no package installation, no configuration differences.

```bash
docker-compose up    # starts FastAPI + Streamlit together
docker-compose down  # stops everything cleanly
```

# Dockerfile — instructions to build the image:

```
# Start with Python 3.10 as our base
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (for faster rebuilds)
COPY requirements.txt .

# Install all Python packages
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy entire project into container
COPY . .

# Create temp folder for image uploads
RUN mkdir -p temp_uploads

# Expose ports for FastAPI and Streamlit
EXPOSE 8000
EXPOSE 8501

# Make the entrypoint script executable
RUN chmod +x entrypoint.sh

# Fixes JSONArgsRecommended warning and maps both processes cleanly
CMD ["./entrypoint.sh"]

```

# docker-compose.yml— runs both services together:

```
services:
  api:            # FastAPI on port 8000
  dashboard:      # Streamlit on port 8501 (waits for API to start)
```

# Without Docker: It works on my machine but not yours.
# With Docker: It works the same way on every machine.


### The Complete Journey at a Glance

```
Step 1 → Define the problem clearly before writing code
Step 2 → Set up a clean, isolated development environment
Step 3 → Explore and understand the data (found class imbalance)
Step 4 → Build and train the CNN model (97.32% test accuracy)
Step 5 → Build the inference pipeline (CNN + Groq AI)
Step 6 → Expose the model as a REST API (FastAPI)
Step 7 → Build the farmguard-ai dashboard (Streamlit)
Step 8 → Write automated tests (5 tests, all passing)
Step 9 → Containerise for deployment (Docker)
```

Each step builds on the previous one. This is the proven, professional order for building an end-to-end ML system — from raw data to a deployed, testable product.

---

## Getting Started

### Prerequisites

- Python 3.10+
- Git
- Docker Desktop (for containerised run only)

### 1. Clone the repository

```b
git clone https://github.com/OmotorMega1/farmguard-ai.git
cd farmguard-ai
```

### 2. Create virtual environment

```
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at: `https://console.groq.com`

### 5. Download the dataset (optional — for retraining only)

Download PlantVillage from Kaggle and place it in `data/raw/PlantVillage/`

---

## Running the App

You need **two terminals** running simultaneously.

### Terminal 1 — Start FastAPI backend

```bash
uvicorn api.main:app --reload
```

API running at: `http://127.0.0.1:8000`

### Terminal 2 — Start Streamlit dashboard

```
streamlit run dashboard/app.py
```

Dashboard running at: `http://localhost:8501`

## API Endpoints

| Method | Endpoint | Description |

| `GET` | `/` | Confirms API is running |
| `GET` | `/health` | Health check — model and API status |
| `POST` | `/predict` | Upload leaf image → returns prediction + AI recommendation |
| `GET` | `/classes` | Lists all 6 supported disease classes |

### Example — POST /predict response

```json
{
  "success": true,
  "disease": "Tomato Early Blight",
  "confidence": "98.85%",
  "severity": "Severe",
  "is_healthy": false,
  "recommendation": "Tomato Early Blight is a fungal disease caused by Alternaria solani..."
}
```

Interactive API docs: `http://127.0.0.1:8000/docs`

---

## Environment Variables

 `GROQ_API_KEY` ✅ Yes | Groq API key for AI treatment generation |

---

## Running Tests

```
python tests/test_predict.py
```

### Expected output

```
==================================================
FARMGUARD AI — TEST SUITE
==================================================
PASSED ✅ — Model Loading
PASSED ✅ — Prediction Structure
PASSED ✅ — All 6 Classes
PASSED ✅ — Healthy Detection
PASSED ✅ — Severity Mapping

5/5 tests passed
🚀 All tests passed — Ready for deployment!
```

---

## Docker

### Build and run

```
docker build -t farmguard-ai
docker-compose up
```

### Stop

```
docker-compose down
```

## Retraining the Model

```bash
python src/train_cnn.py
```

Saves to `models/disease_model.h5`

---

## Supported Crops and Diseases

| Crop | Disease | Healthy 
| 🍅 Tomato | Early Blight | ✅ 
| 🌶️ Pepper | Bacterial Spot | ✅ 
| 🥔 Potato | Early Blight | ✅ 

---

## Acknowledgements

- [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)
- [Groq](https://console.groq.com)
- [Andrew Ng's ML Specialization](https://www.coursera.org/specializations/machine-learning-introduction)


*FarmGuard AI — Helping Nigerian Farmers Protect Their Crops*
