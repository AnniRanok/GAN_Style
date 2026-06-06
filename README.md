# StyleGAN3 Clothing Generation & Inversion

## Overview

This project explores high-resolution image generation and latent space inversion using StyleGAN3 applied to a custom dataset of clothing images.

The system generates realistic flattened clothing representations and investigates inversion techniques to map real-world clothing images back into the latent space.

This is a deep learning research prototype focused on generative modeling and image inversion techniques.


## Objectives

The main goals of this project are:

- Train a generative model for clothing image synthesis
- Evaluate StyleGAN3 performance on a domain-specific dataset
- Explore latent space inversion from real-world images
- Compare domain-specific training with pre-trained face models (FFHQ baseline)

## Dataset

- 1,648 flattened clothing images collected from public sources  
- Preprocessed into TFRecord format for training efficiency  
- Domain-specific dataset focused on clothing textures and shapes  


## Model Architecture

- StyleGAN3 (NVIDIA implementation)
- Trained from scratch on custom clothing dataset
- Resolution: 1024×1024


## Training Setup

- Infrastructure: AWS multi-GPU environment  
- 4 GPUs used for training  
- Batch size: 16  
- Training length: ~500k images (kimg=500)  
- Best checkpoint: `network-snapshot-000432.pkl`  
- Evaluation metric: FID ≈ 31  


## Inversion Methods

The project implements latent space inversion using:

- HyperStyle  
- PTI (Pivotal Tuning Inversion)

These methods are used to map real-world clothing images into the learned latent space for reconstruction and editing.


## Pipeline

1. Data collection and preprocessing  
2. TFRecord conversion  
3. StyleGAN3 training  
4. Image synthesis  
5. Latent space inversion  
6. Evaluation and comparison  


## Results

- The model generates realistic clothing textures at high resolution  
- Domain-specific training improves structural consistency  
- Inversion works reliably but is sensitive to image quality  
- Compared to FFHQ-trained models, clothing domain shows higher variance and lower texture sharpness  


## Limitations

- Dataset size is relatively small for generative modeling  
- Limited diversity in clothing categories  
- Inversion quality depends heavily on input image quality  
- No conditional generation (class-controlled synthesis not implemented)  


## Future Work

- Scale dataset to improve diversity and generalization  
- Introduce conditional StyleGAN (labels for clothing types)  
- Improve inversion accuracy with additional fine-tuning  
- Extend to virtual try-on applications  
- Combine with detection models for end-to-end pipeline  


## Tech Stack

- Python  
- PyTorch  
- NVIDIA StyleGAN3  
- AWS GPU infrastructure  
- HyperStyle (inversion)  
- PTI (Pivotal Tuning Inversion)  


## Status

This project is a deep learning research prototype focused on generative modeling and inversion techniques in a fashion domain.

## Inference

The repository supports image generation and inversion using a pre-trained StyleGAN3 model.

### Generate Images

python stylegan3/gen_images.py \
    --network=weights/stylegan3_model.pt \
    --outdir=generated_clothing \
    --seeds=1-10 \
    --trunc=1.0

### Image Inversion

python scripts/invert.py \
    --input_image input/clothing_photo.jpg \
    --checkpoint_path weights/stylegan3_model.pt \
    --output_path results/

## Notes

- Model performance depends on dataset size and diversity  
- Inversion quality varies depending on input image complexity  
- This is a research prototype focused on GAN behavior in a domain-specific setting  

## Author

Konar Inna  
Machine Learning Engineer & Data Scientist  




