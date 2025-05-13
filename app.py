import streamlit as st
import numpy as np
import tensorflow as tf

# Set up page
st.set_page_config(page_title="Anime Face Generator", layout="centered")

# Custom CSS for background and stylish font
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Poppins', sans-serif;
            background-image: url('https://wallpapers.com/images/featured/cool-anime-background-6kbwj9794wpnsfr1.jpg');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: #ffffff;
        }

        h1 {
            color: #ffffff;
            text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.7);
        }

        .stButton > button {
            background-color: #ff4b6e;
            color: white;
            padding: 0.75em 2em;
            font-size: 1.2em;
            border-radius: 12px;
            border: none;
            transition: background-color 0.3s ease, transform 0.2s ease;
        }

        .stButton > button:hover {
            background-color: #ff2e5c;
            transform: scale(1.05);
        }

        .stSlider label {
            color: #ffffff;
        }

        .css-1v0mbdj, .stSlider {
            backdrop-filter: blur(6px);
            background-color: rgba(0, 0, 0, 0.3);
            border-radius: 12px;
            padding: 1em;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>🎨 Anime Face Generator</h1>", unsafe_allow_html=True)

# Load Generator Model
@st.cache_resource
def load_generator():
    return tf.keras.models.load_model("saved_model/g_model.h5", compile=False)

# Postprocessing function
def postprocess_images(images):
    images = (images + 1) * 127.5
    images = tf.clip_by_value(images, 0, 255)
    images = tf.cast(images, tf.uint8)
    return images.numpy()

# Image count slider
num_faces = st.slider("How many faces do you want to generate?", 1, 25, 5)

# Generate button
if st.button("Generate Face(s)"):
    with st.spinner("🌟 Generating..."):
        g_model = load_generator()
        latent_dim = g_model.input_shape[1]
        noise = np.random.normal(size=(num_faces, latent_dim))
        generated_images = g_model.predict(noise)
        processed_images = postprocess_images(generated_images)

        st.write("### ✨ Your Generated Anime Faces:")
        cols = st.columns(min(num_faces, 5))
        for i, img in enumerate(processed_images):
            cols[i % len(cols)].image(img, use_container_width=True)