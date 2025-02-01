# Smart Waste Classification & Recycling Suggestion System

## Overview
This project implements a deep learning model using MobileNet for smart waste classification. It categorizes waste into different classes using the TrashNet dataset and provides recycling suggestions to promote sustainable waste management.

## Features
- **Real-time Image Classification:** Upload an image, and the model predicts the waste category.
- **Recycling Suggestions:** Get recommendations on how to dispose of or recycle the classified waste.
- **Streamlit Web Application:** A user-friendly interface for easy interaction.

## Dataset
- **Source:** TrashNet Dataset
- **Classes:** Plastic, Metal, Glass, Paper, Cardboard, and Organic Waste
- **Preprocessing:** Image resizing, normalization, and augmentation for better model generalization.

## Model Architecture
- **Base Model:** MobileNet (Pretrained on ImageNet)
- **Fine-tuning:** Last few layers trained on the TrashNet dataset
- **Optimizer:** Adam
- **Loss Function:** Categorical Cross-Entropy
- **Metrics:** Accuracy

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/arpanpramanik2003/smart-waste-classification.git
   cd smart-waste-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run trashnetST.py
   ```

## Usage
1. Open the Streamlit web app.
2. Upload an image of waste.
3. View the predicted category and recycling suggestions.

## Customization
- **Image Size Adjustment:** Ensures that uploaded images appear correctly in the app.
- **Expanded Recycling Information:** Provides more details on how to recycle different waste types.

## Results
- **Training Accuracy:** ~92%
- **Validation Accuracy:** ~88%
- **Loss:** Optimized for minimal classification error

## Deployment
- The model can be deployed on platforms like Render, AWS, or Hugging Face Spaces for online access.

## Contributing
Feel free to open issues and contribute to improving the project.

## License
Apache-2.0 Licence 

## Author
**Arpan Pramanik**

For any queries, reach out via GitHub or LinkedIn.

