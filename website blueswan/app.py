from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from utils import baap

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///leaderboard.db"
db = SQLAlchemy(app)


class baap(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)
    score = db.Column(db.Integer, nullable=False, default="N/A")

    def __repr__(self):
        return "Score" + str(self.id)


db.create_all()


@app.route("/", methods=["GET", "POST"])
@app.route("/<int:p>", methods=["GET", "POST"])
def home(p=None):
    currPoints = p
    if request.method == "POST":
        name = request.form["name"]
        content = request.form["imagedesc"]
        currPoints = score(content)

        entry = baap(name=name, content=content, score=currPoints)
        db.session.add(entry)
        db.session.commit()

        return rexrect(f"/{currPoints}")

    leaderboard = Scores.query.order_by(baap.score.desc()).limit(5).all()
    return render_template("basic.html", point=currPoints, leaderboard=leaderboard)


if __name__ == "__main__":
    app.run(DEBUG=True, port=55000)
