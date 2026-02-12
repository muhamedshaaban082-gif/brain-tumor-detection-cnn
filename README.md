# Brain Tumor MRI Classification 🧠

A deep learning project for classifying brain tumors from MRI images using Convolutional Neural Networks (CNN).

## 📋 Overview

This project implements a CNN-based classifier to detect and classify different types of brain tumors from MRI scans. The model is trained to distinguish between:
- **Glioma**
- **Meningioma**
- **Pituitary tumors**
- **No tumor**

## 🎯 Features

- Multi-class brain tumor classification
- CNN architecture built with TensorFlow/Keras
- Image preprocessing and data augmentation
- Training and testing pipelines
- Performance visualization and evaluation metrics

## 🛠️ Technologies Used

- **Python 3.x**
- **TensorFlow/Keras** - Deep learning framework
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Matplotlib & Seaborn** - Data visualization
- **scikit-learn** - Machine learning utilities
- **PIL** - Image processing

## 📊 Dataset

The dataset contains MRI images organized into:
- **Training set**: 5,712 images
- **Testing set**: 1,311 images

Images are categorized into 4 classes representing different tumor types and healthy brain scans.

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas tensorflow matplotlib seaborn scikit-learn pillow
```

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/brain-tumor-mri-classification.git
cd brain-tumor-mri-classification
```

2. Extract the dataset
```bash
unzip archive.zip -d data/
```

3. Run the Jupyter notebook
```bash
jupyter notebook Brain_Tumor_MRI__1_.ipynb
```

## 📈 Model Architecture

The CNN model consists of:
- Multiple convolutional layers with ReLU activation
- Max pooling layers for dimensionality reduction
- Dropout layers for regularization
- Dense layers for classification
- Softmax output for multi-class prediction

## 🎓 Training

The model uses:
- **Optimizer**: Adam/Adamax
- **Loss Function**: Categorical Crossentropy
- **Data Augmentation**: ImageDataGenerator for better generalization

## 📊 Results

The model achieves competitive accuracy in classifying brain tumor types. Detailed metrics include:
- Confusion Matrix
- Classification Report
- Training/Validation accuracy curves

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.


## 🙏 Acknowledgments

- Dataset providers
- TensorFlow/Keras documentation
- The open-source community

---

**Note**: This project is for educational and research purposes. Medical decisions should always be made by qualified healthcare professionals.
