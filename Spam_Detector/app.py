import pickle
from flask import Flask, render_template, request
import numpy as np

vectorizer = pickle.load(open('spam_tfidf_vect.sav', 'rb'))
model = pickle.load(open('spam_MLP.sav','rb'))

app = Flask(__name__)


@app.route("/")
def homepage():
    return render_template("home.html")


@app.route("/predict", methods=['POST'])
def home():
    msg = request.form['msg']
    msg = vectorizer.transform([msg]).todense()
    y_pred = np.round(model.predict(msg),0)
    return render_template('result.html', data=y_pred)


if __name__ == "__main__":
    app.run(debug=True)