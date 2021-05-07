import pickle
from datetime import timedelta
import numpy as np
import os
from flask import redirect, render_template, session, url_for, request, flash, Flask

app = Flask(__name__)
app.debug = True
app.secret_key = "ZGGL.GBs:M$/DBp"
app.permanent_session_lifetime = timedelta(minutes=5)


@app.route('/')
def home():
    return redirect(url_for('index'))


def predict(values):
    print(os.getcwd())

    with open('rf_model.pkl', 'rb') as f:
        rf = pickle.load(f)

    with open('le.pkl', 'rb') as f:
        le = pickle.load(f)

    with open('std_scalar.pkl', "rb") as f:
        std_scalar = pickle.load(f)

    X = np.array(values).reshape(1, -1)
    X_scaled = std_scalar.transform(X)
    pred = rf.predict(X_scaled)
    return le.inverse_transform(pred)[0]


@app.route('/index', methods=["POST", "GET"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    elif request.method == "POST":
        print("hello")

        input1 = float(request.form["sulphates"])
        input2 = float(request.form["alcohol"])
        input3 = float(request.form["density"])
        input4 = float(request.form["chlorides"])
        input5 = float(request.form["pH"])
        input6 = float(request.form["tso2"])

        values = [input1, input2, input3, input4, input5, input6]

        quality = predict(values)
        print(values)
        print(quality)
        print(f"{input1} {input2} {input3} {input4} {input5} {input6}")

        # for i in values:
        #     if values[i] is None:
        #         flash("Please fill all the values!", "warning")
        #     else:
        #         pass

        return render_template("final.html", quality=quality)


# @app.route('/final')
# def final():
#     # if session['allow']:
#     return render_template('final.html')
