# Smart Waste Classification & Recycling Suggestion System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io/)

## 📖 Overview

A comprehensive deep learning-based solution for intelligent waste classification and recycling recommendations. This system leverages **MobileNet** architecture with transfer learning to accurately categorize different types of waste materials and provide actionable recycling suggestions, contributing to sustainable waste management practices.

The application features a user-friendly web interface built with Streamlit, enabling real-time image classification with instant recycling guidance.

## ✨ Key Features

- **🎯 Accurate Classification**: Deep learning model trained on the TrashNet dataset with high accuracy
- **📸 Real-time Predictions**: Upload waste images and get instant classification results
- **♻️ Recycling Guidance**: Detailed suggestions on proper disposal and recycling methods
- **🖥️ Interactive Web Interface**: Clean, intuitive Streamlit-based UI for seamless user experience
- **🚀 Production-Ready**: Containerized with Docker and deployable on cloud platforms (Fly.io, Heroku)
- **📊 Visual Insights**: Built-in image visualization tools for better understanding

## 🗂️ Project Structure

```
smart-waste-classification/
│
├── app.py                 # Streamlit web application (main entry point)
├── TrashNet.ipynb        # Jupyter notebook for model training and experimentation
├── best_model.keras      # Trained MobileNet model (saved weights)
├── imshow.py             # Image visualization utility functions
│
├── dataset.zip           # TrashNet dataset (compressed)
│
├── requirements.txt      # Python dependencies
├── Dockerfile            # Docker container configuration
├── Procfile              # Heroku deployment configuration
├── fly.toml              # Fly.io deployment configuration
│
├── .gitignore            # Git ignore rules
├── LICENSE               # MIT License
└── README.md             # Project documentation (this file)
```

## 🗄️ Dataset

### TrashNet Dataset

- **Source**: [TrashNet Dataset](https://github.com/garythung/trashnet)
- **Total Images**: 2,527 images
- **Classes**: 6 waste categories
  - 🥤 **Plastic** - Bottles, containers, packaging
  - 🔧 **Metal** - Cans, foils, metal objects
  - 🪟 **Glass** - Bottles, jars, broken glass
  - 📄 **Paper** - Newspapers, magazines, office paper
  - 📦 **Cardboard** - Boxes, cartons, packaging
  - 🍎 **Organic Waste** - Food scraps, biodegradable materials

### Data Preprocessing

- **Image Resizing**: Standardized to 224×224 pixels (MobileNet input size)
- **Normalization**: Pixel values scaled to [0, 1] range
- **Data Augmentation**: 
  - Random rotation
  - Horizontal/vertical flipping
  - Zoom and shift transformations
  - Brightness adjustments
- **Train/Validation Split**: 80/20 ratio

## 🏗️ Model Architecture

### Base Architecture

```
MobileNetV2 (Transfer Learning)
  ├── Input Layer: 224×224×3 RGB images
  ├── MobileNetV2 Base: Pre-trained on ImageNet (frozen layers)
  ├── Global Average Pooling 2D
  ├── Dense Layer (128 units, ReLU activation)
  ├── Dropout (0.5)
  └── Output Layer (6 units, Softmax activation)
```

### Training Configuration

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Fine-tuning Strategy**: Transfer learning with frozen base layers
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Categorical Cross-Entropy
- **Metrics**: Accuracy, Precision, Recall
- **Callbacks**: 
  - ModelCheckpoint (saves best model)
  - EarlyStopping (patience: 10 epochs)
  - ReduceLROnPlateau (learning rate scheduling)
- **Training Epochs**: Variable (with early stopping)
- **Batch Size**: 32

### Model Performance

- **Training Accuracy**: ~95%
- **Validation Accuracy**: ~92%
- **Model Size**: ~14 MB (optimized for deployment)
- **Inference Time**: <100ms per image

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Local Setup

1. **Clone the repository**

```bash
git clone https://github.com/arpanpramanik2003/smart-waste-classification.git
cd smart-waste-classification
```

2. **Create a virtual environment** (recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Extract the dataset** (if needed for training)

```bash
unzip dataset.zip
```

5. **Run the application**

```bash
streamlit run app.py
```

6. **Access the application**

Open your browser and navigate to: `http://localhost:8501`

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t smart-waste-classifier .
```

### Run Docker Container

```bash
docker run -p 8501:8501 smart-waste-classifier
```

### Docker Compose (Optional)

```bash
docker-compose up
```

## ☁️ Cloud Deployment

### Heroku Deployment

1. **Install Heroku CLI**

```bash
heroku login
```

2. **Create Heroku app**

```bash
heroku create your-app-name
```

3. **Deploy**

```bash
git push heroku master
```

### Fly.io Deployment

1. **Install Fly CLI**

```bash
fly auth login
```

2. **Launch app**

```bash
fly launch
```

3. **Deploy**

```bash
fly deploy
```

## 💻 Usage Guide

### Web Application

1. **Launch the application** using one of the methods above
2. **Upload an image** of waste material
   - Supported formats: JPG, JPEG, PNG
   - Recommended: Clear, well-lit images
3. **View prediction results**
   - Waste category classification
   - Confidence score
   - Recycling recommendations
4. **Follow recycling suggestions** for proper waste disposal

### Jupyter Notebook (Training/Experimentation)

```bash
jupyter notebook TrashNet.ipynb
```

Use the notebook to:
- Explore the dataset
- Train new models
- Experiment with different architectures
- Evaluate model performance
- Visualize training metrics

### Image Visualization Utility

```python
from imshow import display_image

# Display image with predictions
display_image('path/to/image.jpg', model, class_names)
```

## 🔧 Configuration & Customization

### Model Customization

Edit `TrashNet.ipynb` to modify:
- Model architecture
- Hyperparameters
- Training strategy
- Data augmentation techniques

### Application Customization

Edit `app.py` to customize:
- UI theme and layout
- Image size limits
- Prediction confidence thresholds
- Recycling suggestion text

### Deployment Configuration

- **Dockerfile**: Modify base image, dependencies, or runtime settings
- **Procfile**: Update Heroku dyno commands
- **fly.toml**: Configure Fly.io deployment settings

## 📦 Dependencies

### Core Dependencies

- **TensorFlow 2.x**: Deep learning framework
- **Keras**: High-level neural networks API
- **Streamlit**: Web application framework
- **NumPy**: Numerical computing
- **Pillow**: Image processing
- **Matplotlib**: Data visualization

### Full Dependencies

See `requirements.txt` for complete list with version specifications.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/YourFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m "Add YourFeature"
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/YourFeature
   ```
5. **Open a Pull Request**

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

## 🐛 Known Issues & Limitations

- Model performance may vary with images taken in poor lighting conditions
- Very small or obscured waste items may be misclassified
- Mixed waste (multiple categories in one image) will classify as the dominant category
- Model size optimized for deployment may have slightly lower accuracy than full model

## 🔮 Future Enhancements

- [ ] Multi-label classification for images containing multiple waste types
- [ ] Real-time video stream classification
- [ ] Mobile application (iOS/Android)
- [ ] Integration with waste management APIs
- [ ] Expanded dataset with more waste categories
- [ ] Localization support for multiple languages
- [ ] User feedback mechanism to improve model
- [ ] Analytics dashboard for waste statistics

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Arpan Pramanik

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## 👨‍💻 Author

**Arpan Pramanik**

- GitHub: [@arpanpramanik2003](https://github.com/arpanpramanik2003)
- Repository: [smart-waste-classification](https://github.com/arpanpramanik2003/smart-waste-classification)

## 🙏 Acknowledgments

- **TrashNet Dataset**: Thanks to Gary Thung and Mindy Yang for creating and sharing the TrashNet dataset
- **MobileNet**: Google Research for the efficient MobileNet architecture
- **Streamlit**: For providing an excellent framework for building ML web applications
- **TensorFlow/Keras**: For powerful deep learning tools and APIs
- **Open Source Community**: For inspiration and various tools that made this project possible

## 📞 Support & Contact

For questions, issues, or suggestions:

- **Open an issue**: [GitHub Issues](https://github.com/arpanpramanik2003/smart-waste-classification/issues)
- **Discussions**: [GitHub Discussions](https://github.com/arpanpramanik2003/smart-waste-classification/discussions)

## 🌟 Star This Repository

If you find this project useful, please consider giving it a ⭐️ on GitHub!

---

**Made with ❤️ for a sustainable future**
