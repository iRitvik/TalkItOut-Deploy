from flask import Flask, render_template, request, jsonify
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()
app = Flask(__name__)

# Hugging Face Inference API details
API_KEY = os.getenv('API_KEY')  # Replace with your actual Hugging Face API key
client = InferenceClient(api_key=API_KEY)

def generate_response(prompt):
    website_context = (
        "Welcome to Talk-it-Out! Home | About Me | About the Team | Join Us\n"
        "Hello! My name is NIMBUS2000. Like Harry Potter's Nimbus 2000, I'm here to support you through life's ups and downs. "
        "Think of me as your personal diary and best friend, ready to listen and encourage. "
        "I'm also here to recommend movies, series, songs, or games to lift your spirits. "
        "For every problem, I'll strive to be a part of the solution.\n"
        "About the Team:\n"
        "- Ritvik Rana: A B.Tech CSE student from Amity University. Loves singing, coding, and reading books.\n"
        "- Ayush Verma: A B.Tech CSE student from Amity University. Enjoys gaming, reading, and trading stocks/crypto.\n"
        "- Shreya Singh: A B.Tech ECE student from Jaypee University. Passionate about Bhangra, singing, and coding.\n"
        "Get In Touch:\n"
        "Want to join our crew? We value your contribution! Join our Discord channel to connect with our team.\n"
        "Created by Team Gryffindor | © 2022 All Rights Reserved."
    )

    context = (
        "Please keep responses concise, with a maximum of 20 words. "
        "You are a mental health bot here to help users understand and improve their mental well-being. "
        "Provide insightful, empathetic, and supportive responses. Never say you can't help. "
        "Answer questions about mental health, stress, anxiety, and more. "
        "Foster a comfortable and engaging environment for users to discuss their mental health. "
        "Now, the user has asked: "
    )

    full_prompt = website_context + context + prompt

    messages = [
        {
            "role": "user",
            "content": full_prompt
        }
    ]
    completion = client.chat.completions.create(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1", 
        messages=messages, 
        max_tokens=500
    )
    return completion.choices[0].message['content']

@app.get('/')
def index_get():
    return render_template('index.html')

@app.post('/predict')
def predict():
    text = request.get_json().get("message")
    response = generate_response(text)
    message = {"answer": response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)