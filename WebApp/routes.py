from flask import Flask, redirect, url_for, render_template, request

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def index():
    if request.method == "POST":
        seed = request.form["nm"]
        generatedText = seed
        if len(seed) != 0:
        	return redirect(f"/generate/{generatedText}")
        else:
            return redirect("/generate/please enter a valid seed")
    else:
        return render_template("landingpage.html")


@app.route('/generate/<lyrics>')
def generate(lyrics):
	arr = lyrics.split(' ')
	return render_template('result.html',lyrics=arr[0])


if __name__ == "__main__":
    app.run(debug=True)