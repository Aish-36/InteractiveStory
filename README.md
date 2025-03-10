📖 Interactive Storytelling Platform for Hearing Impaired
📌 Project Overview
This project is an interactive storytelling platform designed for children with hearing impairment. It converts text into sign language videos, using a dataset of ASL images, GIFs, and pre-recorded sign videos. It also generates audio narration for better accessibility.

📂 Repository Structure
📂 Project Root
│── 📂 src # Source code
│ ├── preprocessing.py # Data preprocessing
│ ├── train.py # Model training
│ ├── evaluate.py # Model evaluation
│ ├── app.py # Deployment (Flask/Streamlit)
│
│── 📂 notebooks # Jupyter notebooks
│
│── 📂 data # Dataset files
│ ├── 📂 asl_dataset # ASL alphabets
│ ├── 📂 images # Sign language images
│ ├── 📂 gifs # Sign language GIFs
│ ├── 📂 temp # Audio files for generated text
│ ├── 📂 videos # Sign videos for custom sentences
│
│── 📂 models # Trained models
│
│── 📂 results # Logs & outputs
│
│── 📂 docs # Documentation
│
│── README.md # Project overview
│── requirements.txt # Dependencies
│── .gitignore # Ignore unnecessary files
│── .env # API keys (should be ignored in Git)