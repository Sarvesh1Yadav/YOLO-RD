# Research Implementation: YOLO-rd with VM/VLM/LLM Initializers on COCO Subset

## 📌 Introduction
This repository contains my research-based implementation of the **YOLO rd-9c architecture**, inspired by recent advancements in object detection.  
The work focuses on experimenting with **different initializers** (Vision-based, Vision-Language-based, and Language-based) to analyze their effect on **object detection performance (mAP)** using a **subset of the COCO dataset**.  

The project is implemented, trained, and benchmarked on **Kaggle GPU accelerators**.

---

## 📖 Abstract
Object detection has evolved significantly with the YOLO family of models.  
In this study, I reproduced and fine-tuned the **YOLO rd-9c model** with **different weight initializations**:
- **VM (Vision Model initializer)**  
- **VLM (Vision-Language Model initializer)**  
- **LLM (Language Model initializer)**  

The primary goal was to test whether **cross-modal initializations** (VLM/LLM) improve transferability and detection performance compared to standard vision-based initialization.  

---

## ⚙️ Methodology

### 🔹 Model
- **Base architecture**: `rd-9c.yaml`  
- **Custom modules**: RepNCSPELAN, ADown, CBLinear, MultiheadDetection  
- **Checkpoints used**:
  - `rd-9c.pt` → Standard VM initializer  
  - `rd-9c-vlm.pt` → VLM-initialized using CLIP vision embeddings  
  - `rd-9c-4096.pt` → LLM-initialized using GPT-2  

### 🔹 Dataset
- **COCO Dataset Subset**: 40 classes out of 80  
- Training and validation splits prepared with COCO-style annotations  
- Only selected categories were used for efficient training & evaluation  

### 🔹 Training
- Environment: Kaggle Notebook (A100 GPU / T4 GPU)  
- Training command:
  ```bash
  yolo train model=rd-9c.yaml data=coco-subset.yaml epochs=100 imgsz=640 batch=16

## 📁 Folder Structure

YOLO-RD/
├── .github/                 
├── .vscode/                
├── demo/                 
├── docker/                 
├── docs/                   
├── examples/               
├── tests/               
├── yolo/                   
├── my_project_root.code-workspace 
├── .gitignore              
├── .gitattributes
├── .pre-commit-config.yaml
├── .readthedocs.yaml
├── LICENSE
├── README.md
├── requirements.txt       



---

## 📦 Setup Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/Sarvesh1Yadav/YOLO-RD.git
   cd YOLO-RD

