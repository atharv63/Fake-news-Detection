from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W',' ',text)
    text = re.sub(r'\s+',' ',text)
    return text


@app.route("/")
def home():
    return render_template("index.html", prediction=None)


@app.route("/predict", methods=["POST"])
def predict():

    title = request.form["title"]
    news = request.form["news"]

    content = title + " " + news
    content = clean_text(content)

    vector = vectorizer.transform([content])

    prediction = model.predict(vector)

    if prediction[0] == 0:
        result = "Fake News"
    else:
        result = "Real News"

    return render_template(
        "index.html",
        prediction=result,
        title=title,
        news=news
    )


if __name__ == "__main__":
    app.run(debug=True)