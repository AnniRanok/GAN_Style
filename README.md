
# StyleGAN3: High-Resolution Generative Modeling & Latent Space Inversion

## Overview
This project explores high-resolution generative modeling and latent space inversion using StyleGAN3 trained on a custom dataset.  
The focus is on end-to-end ML pipeline development, large-scale model training, and evaluation of representation learning under limited-data conditions.

The project emphasizes training stability, data pipeline design, and GPU-based deep learning experimentation.


## Key Objectives
- Train a high-resolution generative model using StyleGAN3 from scratch  
- Build a scalable data preprocessing and training pipeline  
- Evaluate generative performance under small dataset constraints  
- Implement latent space inversion for reconstructing real images  
- Analyze limitations of GAN training in low-data regimes  


## Dataset
- 1,648 domain-specific images (clothing dataset)  
- Preprocessed into TFRecord format for efficient training  
- Focus on structured visual data with limited sample diversity  


## Model Architecture
- NVIDIA StyleGAN3 (unconditional GAN)  
- Resolution: 1024×1024  
- Training from scratch on custom dataset  


## Training Infrastructure
- AWS multi-GPU environment (4 GPUs)  
- Batch size: 16  
- Total training scale: ~500k images (kimg = 500)  
- Checkpoint: `network-snapshot-000432.pkl`  


## Evaluation
- Fréchet Inception Distance (FID): ~31  
- Observed trade-off between diversity and texture sharpness due to dataset scale  
- Stable convergence achieved under constrained data regime  


## Latent Space Inversion
Implemented inversion techniques:
- HyperStyle  
- PTI (Pivotal Tuning Inversion)  

These methods enable mapping real images into the GAN latent space for reconstruction analysis.


## Pipeline
- Data collection and preprocessing  
- TFRecord conversion for optimized training  
- StyleGAN3 training pipeline  
- Image synthesis  
- Latent space inversion  
- Model evaluation  


## Key Learnings
- Challenges of training deep generative models on small datasets  
- Sensitivity of GANs to data distribution and diversity  
- Trade-offs between fidelity and generalization  
- Practical constraints of multi-GPU training pipelines  
- Importance of preprocessing and data representation quality  


## Limitations
- Limited dataset size affects generalization performance  
- No conditional generation (no label-driven control)  
- Inversion quality depends heavily on input image quality  
- Domain is not optimized for downstream predictive tasks  


## Tech Stack
Python • PyTorch • NVIDIA StyleGAN3 • AWS (EC2 GPU) • TFRecord  
HyperStyle • PTI  


## Inference

### Image Generation

python stylegan3/gen_images.py \
  --network=weights/stylegan3_model.pt \
  --outdir=generated \
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

## Author

Konar Inna  
Machine Learning Engineer & Data Scientist  




### Image Inversion

python scripts/invert.py \
    --input_image input/clothing_photo.jpg \
    --checkpoint_path weights/stylegan3_model.pt \
    --output_path results/

## Notes

- Model performance depends on dataset size and diversity  
- Inversion quality varies depending on input image complexity  

## Author

Konar Inna  
Machine Learning Engineer & Data Scientist  




