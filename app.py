# ChatBot-PretrainedModel
# Author: Edgar Rodriguez
# ChatBot example with GPT-2 use a pretrained model on very large corpus of English language 
# using a causal language modeling (CLM).

# The author generated this text in part using GPT-3, OpenAI's large-scale language generation model.
# After generating the draft of the text, the author reviewed, edited, revised and test it.
# This project is licensed under the terms of the MIT License

from flask import Flask, request, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.form["user_input"]
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    response_ids = model.generate(input_ids, max_length=50, num_return_sequences=1)
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    app.run(debug=True)
