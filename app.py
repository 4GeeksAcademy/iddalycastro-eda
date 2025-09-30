from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

MODEL_PATH = "model_app.pkl"
model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    inputs = {}
    error = None

    if request.method == "POST":
        try:
            duration = float(request.form.get("duration", "").strip())
            followers = float(request.form.get("followers", "").strip())
            live_status = request.form.get("live_status", "False").strip()

            X = [{"duration": duration,
                  "channel_follower_count": followers,
                  "live_status": live_status}]
            y_pred = model.predict(X)[0]
            prediction = int(max(y_pred, 0))
            inputs = {"duration": duration, "followers": followers, "live_status": live_status}
        except Exception as e:
            error = str(e)

    return render_template("index.html", prediction=prediction, inputs=inputs, error=error)

if __name__ == "__main__":
    app.run(debug=True)
