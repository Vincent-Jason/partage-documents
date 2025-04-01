from flask import Flask, request, render_template
from transformers import pipeline

app = Flask(__name__)

model_id = "google/gemma-3-1b-it"

# Charger le modèle sur CPU avec des paramètres optimisés
generator = pipeline("text-generation", model=model_id, device="cpu")

def generate_description(keywords):
    prompt = f"Rédige une description optimisée pour le SEO sur : {keywords}. La description doit être claire, attractive et naturelle."
    result = generator(prompt, max_length=150, num_return_sequences=1, do_sample=True, temperature=0.7, repetition_penalty=1.2)
    return result[0]['generated_text']

@app.route("/", methods=["GET", "POST"])
def home():
    description = None
    if request.method == "POST":
        keywords = request.form["keywords"]
        description = generate_description(keywords)
    return render_template("index.html", description=description)

if __name__ == "__main__":
    app.run(debug=True)
