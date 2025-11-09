# 🧠 Deepfake Detection – Synergy’25 ML Hackathon

## 📘 Overview
This repository contains the submission for **Face the Future: Deepfake ML Challenge (Synergy’25)**.  
The project detects manipulated facial media using deep learning. It integrates **CNN**, **YOLOv8**, and **transformer-based** architectures to achieve robust detection results.

---

## 🚀 Features
- Full ML pipeline for training, evaluation, and inference  
- Image preprocessing and face alignment  
- JSON-based output for hackathon submission  
- Streamlit-based demo app for testing  
- Scalable design with blockchain-ready verification layer  

---
deepfake-blockchain/
│
├── streamlit_interface.py # Streamlit app
├── final_contract.sol # Solidity smart contract
├── ABI_DeepfakeLedger.json # ABI exported from Remix
├── best_deepfake_model_94.75.pth # Pretrained ResNet-18 model
├── requirements.txt # Python dependencies
└── README.md # Documentation
└── best_deepfake_model_94.5.pth # Pretrained ResNet-18 model No. 2
└──test1/ #folder including test images with zip file of all images
---
## ⚙️ Installation
Clone and install:
```bash
git clone https://github.com/<your-github-username>/deepfake-ml-challenge.git
cd deepfake-ml-challenge
pip install -r requirements.txt
🌐 Blockchain Setup
Install Tools
✅ Ganache

Download: https://trufflesuite.com/ganache/

Start Ganache and note:

RPC Server → HTTP://127.0.0.1:7545

Accounts + private keys (for Streamlit selection)

✅ Remix IDE

Use online IDE: https://remix.ethereum.org/
Deploy the Smart Contract

Open Remix IDE

Open file → final_contract.sol

Then compile the file and deploy on Ganache server HTTP://127.0.0.1:7545

---
💻 Run the Streamlit App
 Launch
streamlit run streamlit_interface.py