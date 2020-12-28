from flask import Flask, redirect, url_for, render_template, request, jsonify
from WebApp.model import Generator

model = Generator()

model.load_weights('model-1-epochs-256-neurons.h5')


app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def index():
    if request.method == "POST":
        seed = request.form["nm"]
        generatedText = model.predict(start_seed=seed, gen_size=1000, temp=1.0)
        if len(seed) != 0:
        	return redirect(f"/generate/{generatedText}")
        else:
            return redirect("/generate/please enter a valid seed")
    else:
        return render_template("landingpage.html")


@app.route('/generate/<lyrics>')
def generate(lyrics):
    lyricsSplit = lyrics.split('\n')
    return render_template('result.html',lyrics=lyricsSplit)

@app.route('/api/generate/<seed>')
def api(seed):
    generatedText = model.predict(start_seed=seed, gen_size=1000, temp=1.0)
    return jsonify({'lyrics' : generatedText})
