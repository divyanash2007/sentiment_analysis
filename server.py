from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# --------------------------
# Load your trained model
# --------------------------
model = pickle.load(open("./notebook/model.pkl", "rb"))
vectorizer = pickle.load(open("notebook/vectorise.pkl", "rb"))
# Make sure you saved both during training


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    review = request.form.get("review")

    # Transform the review using same TF-IDF
    transformed = vectorizer.transform([review])

    # Predict
    pred = "positive" if model.predict(transformed) == 1 else "negative"

    # For probability (optional)
    # prob = model.predict_proba(transformed)[0]

    # Render result back on page
    return render_template("index.html", sentiment=pred)


if __name__ == "__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0",port=port)