
# **StyleGAN3 Clothing Generation & Inversion**  

# **Project Overview**  
This project leverages **StyleGAN3** to generate **realistic flattened clothing images** and perform **image inversion** for further editing. The model was trained on a **custom dataset of 1,648 flattened clothing images**, collected from the Internet.  

The goal is to enable **high-quality clothing synthesis** while allowing inversion from real-world images of people wearing clothes back into the flattened clothing representation.  

---

# **Key Features**  
 **StyleGAN3 Training on Flattened Clothing** – A custom model trained from scratch to generate flattened clothing images.  
 **Inversion from Real-World Images** – Mapping worn clothing to its flattened version using **latent space inversion**.  
 **High-Resolution Outputs** – The model generates **1024×1024** images with realistic textures and details.  
 **Comparison with Large-Scale Training on Faces** – Demonstrates the maximum achievable quality using a **pre-trained StyleGAN3 model on FFHQ (faces dataset)**.  
 **AWS Model Training** – Final model training was conducted on **AWS (4 GPUs, batch size 16, 500k images, 20 hours training time)**.  

---

# **Project Pipeline**  

### ** Dataset Collection & Processing**  
- **Collected 1,648 flattened clothing images** from various sources.  
- Preprocessed images into **TFRecord format** for efficient training.  

### ** Model Training**  
- Initially attempted training on **Kaggle**, but was limited by resources.  
- Final training took place on **AWS** using:  
  - **4 GPUs**  
  - **Batch size: 16**  
  - **Training steps: 500k images (kimg=500)**  
  - **Best model: network-snapshot-000432.pkl (FID = 31)**  

### ** Image Generation**  
- Generated **10+ clothing images** using trained weights.  
- Compared results with a **StyleGAN3 model trained on FFHQ (faces dataset)** to demonstrate achievable accuracy.  

### ** Inversion & Editing**  
- Implemented **latent space inversion** to reconstruct clothing from real-world images.  
- Used **HyperStyle & PTI (Pivotal Tuning Inversion)** for improved inversion accuracy.  

---

##  **How to Run the Project**  

### ** Clone the Repository**  
```bash
git clone https://github.com/your-repo/stylegan3-clothing.git
cd stylegan3-clothing

# Install Dependencies
pip install -r requirements.txt

# Generate New Clothing Images

python stylegan3/gen_images.py \
    --network=weights/stylegan3_model.pt \
    --outdir=generated_clothing \
    --seeds=1-10 \
    --trunc=1.0

# Perform Inversion

python scripts/invert.py \
    --input_image input/clothing_photo.jpg \
    --checkpoint_path weights/stylegan3_model.pt \
    --output_path results/

#  Results & Observations

-  Training on limited clothing data resulted in good generation quality, but more data would improve diversity.
-  The inversion process successfully reconstructs clothing, but further fine-tuning is needed for best accuracy.
-  Compared to a StyleGAN3 model trained on faces, the clothing model has a lower level of detail but still achieves realistic results.

#  Future Improvements

-  Increase dataset size – More training images for better diversity.
-  Train separate models for clothing & shoes – Avoid classification errors.
-  Improve inversion accuracy – Use PTI fine-tuning for better latent space mapping.
-  Implement virtual try-on system – Integrate generated clothing into a virtual wardrobe.

#  References

StyleGAN3 Paper
Official NVIDIA Implementation
HyperStyle for Inversion
PTI for High-Quality Inversion

# Author
Konar Inna – Machine Learning Engineer & Data Scientist
 konar.inna@gmail.com
