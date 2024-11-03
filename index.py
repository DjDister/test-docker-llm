from flask import Flask, request, jsonify
from transformers import pipeline

# Initialize the model
pipe = pipeline("text-generation", model="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF")

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate_text():
    user_input = request.json['input']
    result = pipe([{"role": "user", "content": user_input}])
    return jsonify(result[0]['generated_text'])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)