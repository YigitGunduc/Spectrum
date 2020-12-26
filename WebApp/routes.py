from flask import Flask, redirect, url_for, render_template, request, jsonify
from model import Generator


model = Generator()

model.load_weights('model-0-epochs.h5', mode = 'predition')

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
    return render_template('result.html',lyrics=lyrics)

@app.route('/api/generate/<seed>')
def api(seed):
    generatedText = generate_text(model, seed,gen_size=700,temp=1.0)
    return jsonify({'lyrics' : generatedText})

if __name__ == "__main__":
    app.run(debug=True)
