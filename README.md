```md
# Cat-Dog Classifier (ResNet18 + FastAPI + Frontend)

A full-stack image classification project that predicts whether an uploaded image contains a cat or dog using a fine-tuned ResNet18 model. The backend is built with FastAPI, and the frontend uses plain HTML, CSS, and JavaScript.

---

## Features

- ResNet18-based image classifier
- FastAPI REST API backend
- Image upload and prediction endpoint
- Probability/confidence output
- Lightweight frontend with no framework
- Separated training, inference, and UI structure

---

## Project Structure

```text
Cat-Dog-Classifier/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── resnet_model.py
│   │   └── __init__.py
│   │
│   ├── models/
│   │   └── resnet18_catdog.pth
│   │
│   ├── training/
│   │   └── resnet18_cat_dog.py
│   │
│   └── requirements.txt
│
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── script.js
│
├── README.md
└── .gitignore
```

---

## Backend Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Run the API Server

```bash
uvicorn app.main:app --reload
```

### 3. Available Endpoints

| Method | Endpoint | Description           |
| ------ | -------- | --------------------- |
| GET    | /        | API status check      |
| GET    | /health  | Device and model info |
| POST   | /predict | Predict cat or dog    |

### 4. Example Prediction Response

```json
{
  "prediction": "dog",
  "confidence": 0.9821
}
```

---

## Frontend Setup

Run a local static server:

```bash
cd frontend
python -m http.server 5500
```

Open in browser:

```text
http://localhost:5500
```

---

## Model Details

* Architecture: ResNet18
* Pretrained on ImageNet
* Final layer replaced for 2 classes
* Classes:

  * cat
  * dog
* Input image size: 224 x 224
* Loss function: CrossEntropyLoss
* Optimizer: Adam

---

## Training

To retrain the model:

```bash
cd backend/training
python resnet18_cat_dog.py
```

Weights are saved to:

```text
backend/models/resnet18_catdog.pth
```

---

## API Example

Using curl:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@image.jpg"
```

---

## Notes

* Ensure the backend server is running before using the frontend
* Ensure the model file exists in `backend/models/`
* CORS is open for development and should be restricted in production

---

## Tech Stack

* Python
* PyTorch
* TorchVision
* FastAPI
* HTML
* CSS
* JavaScript

---

## Future Improvements

* Docker deployment
* React frontend
* Better UI/UX
* Batch prediction API
* Cloud deployment
* Model monitoring
