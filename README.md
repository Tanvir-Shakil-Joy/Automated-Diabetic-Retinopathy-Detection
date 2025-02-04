# Automated Diabetic Retinopathy Detection

A deep learning-based web application for automated detection and severity classification of Diabetic Retinopathy (DR) from retinal fundus images.

## Features

- Binary classification of DR (Present/Absent)
- Multi-class severity classification (Mild/Moderate/Severe/Proliferative)
- Real-time image processing and prediction
- User-friendly web interface
- Detailed prediction confidence scores

## Tech Stack

- **Backend**: Django
- **Deep Learning**: TensorFlow
- **Image Processing**: OpenCV
- **Frontend**: HTML/CSS/JavaScript
- **Models**: 
  - Ensemble model with SE-Attention for binary classification
  - Multi-class ensemble model for severity detection

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Tanvir-Shakil-Joy/Automated-Diabetic-Retinopathy-Detection.git
cd Automated-Diabetic-Retinopathy-Detection
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # For Unix/macOS
venv\Scripts\activate     # For Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download model files:
- Place `ensemble_se_attention_model.h5` in `predictor/model/`
- Place `ensemble_attention_model_severity.h5` in `predictor/model/`

5. Run migrations:
```bash
python manage.py migrate
```

6. Start the development server:
```bash
python manage.py runserver
```

## Usage

1. Access the web interface at `http://localhost:8000`
2. Upload a retinal fundus image
3. Get instant predictions for:
   - Presence of DR
   - Severity classification (if DR is detected)
   - Confidence scores

## Model Architecture

- **Binary Classification**: Ensemble model combining ResNet50 and VGG16 with Squeeze-and-Excitation attention
- **Severity Classification**: Multi-class ensemble model with attention mechanisms

## Image Preprocessing

- Green channel extraction
- Gaussian blur for noise reduction
- CLAHE for contrast enhancement
- Circular masking
- Gamma correction
- Normalization

## Contributors

- Tanvir Shakil Joy

## License

This project is licensed under the MIT License - see the LICENSE file for details.
