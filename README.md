# 📱 App Review RAG System

**Course:** Natural Language Processing CS 6120  
**Project:** Final RAG System Demo

This project is a Retrieval Augmented Generation (RAG) system that runs entirely on your own computer. It searches through a large database of over 10,000 app reviews to answer user questions. It uses local machine learning models (Mistral via Ollama, SentenceTransformers, and BM25) to write helpful bug reports and provide clickable sources for its answers.

## ✨ Features
1. **💬 Interactive Q&A**: Ask questions about app reviews. The system searches for the 5 most relevant reviews and uses them to answer your question. It provides clickable links, like `[Review #1234]`, so you can read the exact review it used for the answer.
2. **📊 Global Dataset Analysis**: This tab automatically creates summary reports for common issues, like battery drain or app crashes. It searches for the top 5 reviews for each topic, reads them, and writes a professional report.

## 🛠 What You Need Installed
Because this project runs locally to keep your private data safe, you need to install these two programs:
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) (to run the app easily)
* [Ollama](https://ollama.com/) (to run the Mistral AI model)

## 🚀 How to Run the App (Using Docker)

We use Docker so anyone can easily run this app without installing complex Python code directly.

### Step 1: Start the Local AI Server
First, make sure Ollama is downloaded. Since Docker runs in its own environment, we need to tell Ollama to allow Docker to talk to it.

**On your Mac Terminal:**
Open a simple terminal window and run this command:
```bash
OLLAMA_HOST="0.0.0.0" ollama serve
```
Leave this terminal window open in the background!

**In a second, new terminal window**, download and start the AI model:
```bash
ollama run mistral
```

### Step 2: Build the Docker App
Navigate to the main folder of this project in your terminal and build the app:
```bash
docker build -t rag-app-review .
```

### Step 3: Run the Docker App
Start the app using this command:
```bash
docker run -p 8501:8501 rag-app-review
```
It will take a few seconds to load the search models into memory.

### Step 4: Open the App
Go to your web browser and open this link:
```text
http://localhost:8501
```

## 🏗 How It Works Behind the Scenes
* **User Interface**: Built with Streamlit
* **Local APIs**: Uses Docker to safely talk to Ollama (`http://host.docker.internal:11434`)
* **Search System**: Uses BM25 (keyword search) and SentenceTransformer (meaning search) to find the best reviews.
* **AI Model**: Mistral (via Ollama)

## ⚖️ Meeting the Class Guidelines
- **Local Models:** The entire system runs locally on your computer. It does not use any cloud APIs like OpenAI.
- **Easy Setup:** The system is completely containerized with Docker, meaning the instructors can run it perfectly without dealing with Python libraries.
- **Clickable Citations:** The AI provides exact citations for all answers. The app shows the original review text right below the answer to prove the AI did not hallucinate.
