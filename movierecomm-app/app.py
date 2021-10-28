from flask import Flask, render_template, request
#from nmf_recommender import get_recommendation
from cs_recommender import get_recommendation


app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template("main.html", title = "Kickass Movie Recommender")


@app.route('/recommendations')

def recommender():
    form_data = dict(request.args)
    UserX_recommendations = get_recommendation(form_data)

    return render_template("recommendations.html",
    title="Get Recommendations",
    recommendations=UserX_recommendations)

if __name__=="__main__":
    app.run(debug=True, port=5000)
