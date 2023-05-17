from flask import Flask
import pandas as pd
from random import uniform

df = pd.read_csv("../df_clean.csv")
df_sample = df.sample(50)
df = df.set_index("SK_ID_CURR") 

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/df")
def dataframe():
    return df_sample.to_dict()

@app.route("/get_ID_client")
def get_ID_client():
    return list(df_sample["SK_ID_CURR"])

@app.route("/get_info_client/<id_client>")
def get_info_client(id_client):
    info_client = df.loc[int(id_client)]
    return info_client.to_dict()

@app.route("/predict_solvabilite/<id_client>")
def predict_solvabilite(id_client):
    solvabilite = df.loc[int(id_client)]
    return str(uniform(0,1))

@app.route('/user/<username>')
def show_user_profile(username):
    # show the user profile for that user
    return f'User {escape(username)}'

@app.route('/post/<int:post_id>')
def show_post(post_id):
    # show the post with the given id, the id is an integer
    return f'Post {post_id}'
app.run(port=8081)