# 🛡️ Hate Speech Detection Model

This repository contains the code for a **Hate Speech Detection Model** using **TinyBERT**. The model is trained to classify text into the following categories:
- **Normal**
- **Hate Speech**
- **Offensive Language**

## 🚀 How to Use

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/priyanshu596/hate_speech_detection.git
cd hate_speech_detection
```

### 2️⃣ Install Dependencies
Since there is no `requirements.txt`, install dependencies manually:
```bash
pip install transformers torch streamlit
```

### 3️⃣ Download the Model
The trained model is hosted on **Hugging Face Hub**. You can download and use it as follows:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "priyanshu201/Hate_speech_detection"  # Change if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
```
Alternatively, manually download the model from:
[🔗 Hugging Face Model](https://huggingface.co/priyanshu201/Hate_speech_detection)

### 4️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

## 📂 Repository Files
- `app.py` - Streamlit app for hate speech detection
- `data_preparation.ipynb` - Jupyter Notebook for training and preprocessing
- `dataset.json` - Preprocessed dataset used for training

## 🏆 Features
✅ Detects **Hate Speech, Toxic Content, and Offensive Language**  
✅ Based on **TinyBERT** for **lightweight and fast inference**  
✅ Deployable with **Streamlit**

## 📌 Example Usage
You can test the model in Python as follows:
```python
text = "I hate you!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
print(outputs.logits)
```

## 🔗 Useful Links
- **Hugging Face Model:** [Download Here](https://huggingface.co/priyanshu201/Hate_speech_detection)
- **Live App:** [Try Here](https://huggingface.co/spaces/priyanshu201/hate_speech)

## 👨‍💻 Author
**Priyanshu Singh**  
📧 Contact: *your_email@example.com*  
🔗 [LinkedIn](#) | [GitHub](https://github.com/priyanshu596) | [Hugging Face](https://huggingface.co/priyanshu201)

---
Let me know if you need any modifications! 🚀

