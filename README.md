

```markdown
# 🛡️ Hate Speech Detection Model

This repository contains the code for a **Hate Speech Detection Model** using **TinyBERT**. The model is trained to classify text into the following categories:
- **Normal**
- **Hate Speech**
- **Offensive Language**

## 🚀 How to Use

### 1️⃣ Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/priyanshu596/hate_speech_detection.git
cd hate_speech_detection
```

### 2️⃣ Install Dependencies
First, install all required dependencies:
```bash
pip install -r requirements.txt
```
If you prefer to install dependencies manually, use the following:
```bash
pip install transformers torch streamlit pandas numpy nltk spacy
```

### 3️⃣ Download the Model
The trained model files are hosted on **Google Drive**. You can download them manually:
- **Model Weights** (`model.safetensors`) and **Configuration File**: [Download Here](https://drive.google.com/drive/folders/1nDcc6Bgt7tMa-BqGNLtNmKlDuDO09xzz?usp=drive_link)

After downloading, load the model in your code as follows:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Provide the local path to your model and config files
model_path = "<path-to-model>"
config_path = "<path-to-config>"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config_path)
```

### 4️⃣ Run the Streamlit App
To run the app and use the model for predictions, use the following command:
```bash
streamlit run app.py
```

## 📂 Repository Files
- `app.py` - Streamlit app for hate speech detection
- `dataset_preparation.ipynb` - Jupyter Notebook for training and preprocessing the dataset
- `dataset.json` - Preprocessed dataset used for training
- `requirements.txt` - List of dependencies required to run the project

## 🏆 Features
- ✅ Detects **Hate Speech**, **Toxic Content**, and **Offensive Language**
- ✅ Based on **TinyBERT** for **lightweight and fast inference**
- ✅ Deployable with **Streamlit**

## 📌 Example Usage
You can test the model in Python as follows:
```python
text = "I hate you!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
print(outputs.logits)
```

## 🔗 Useful Links
- **Hugging Face Model**: [Download Here](https://huggingface.co/priyanshu201/Hate_speech_detection)
- **Live App**: [Try Here](https://huggingface.co/spaces/priyanshu201/hate_speech)

## 👨‍💻 Author
**Priyanshu Singh**  
📧 Contact: priyanshu.asn2003@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/priyanshu-singh-22b4a3207/) | [GitHub](https://github.com/priyanshu596) | [Hugging Face](https://huggingface.co/priyanshu201)

