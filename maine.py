from flask import Flask, request, render_template
import torch
from diffusers import StableDiffusionPipeline

app = Flask(__name__)
print(__name__)


# Charger le modèle Stable Diffusion (CPU ou GPU si disponible)
model_id = "CompVis/stable-diffusion-v1-4"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to("cpu")  # Remplace par "cuda" si tu as une carte graphique compatible
print("Modèle chargé.")

def generate_image(prompt):
    """Génère une image à partir d'un super texte et la sauvegarde."""
    image = pipe(prompt, num_inference_steps=50).images[0]
    image_path = "static/generated_image.png"
    image.save(image_path)
    return image_path

@app.route("/", methods=["GET", "POST"])
def home():
    print("Génération de la route API.")
    image_path = None
    if request.method == "POST":
        prompt = request.form["prompt"]
        image_path = generate_image(prompt)
    return render_template("indeux.html", image_path=image_path)

if __name__ == "__main__":
    print("L'application démarre...")
    app.run(debug=True)

