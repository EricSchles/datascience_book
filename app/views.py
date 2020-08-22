from app import app

@app.route("/", methods=["GET", "POST"])
def index():
    return "hi"

