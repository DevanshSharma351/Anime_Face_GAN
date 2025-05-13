# 🎨 Anime Face Generator using GAN

This project is a Generative Adversarial Network (GAN) that creates anime-style face images. It uses a deep learning model trained on a dataset of anime faces to generate new, realistic-looking images from random noise vectors.

---

## 🚀 Live Demo

👉 **Streamlit App:** *[Coming Soon – add your Streamlit Cloud link here]*

---


## 🗂️ Project Structure

```
Anime-GAN-App/
├── app.py               # Streamlit web app to generate and display anime faces
├── gan.py               # Script to train the GAN model
├── test.py              # Script to test the generator and save sample images
├── requirements.txt     # List of dependencies
├── saved_model/
│   ├── g_model.h5       # Trained generator model
│   └── d_model.h5       # Trained discriminator model
├── fake.png             # Sample output image
└── README.md            # Project description and instructions
```

---

## 🧠 Model Overview

- **Architecture**: Standard GAN with separate Generator and Discriminator models.
- **Latent Space**: 128-dimensional Gaussian noise vector.
- **Generator**: Deep neural network using LeakyReLU and Tanh activations.
- **Discriminator**: Classifier using LeakyReLU and Sigmoid activations.
- **Optimizer**: Adam (`learning_rate = 0.0002`, `beta_1 = 0.5`)
- **Loss Function**: Binary Crossentropy
- **Training Dataset**: Anime Faces Dataset from Kaggle

---

## 🖼️ Generated Sample

![Generated Sample](/fake.png)

---