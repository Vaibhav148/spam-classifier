import pandas as pd
from flask import Flask, jsonify, request,json
import pickle

# load model
model = pickle.load(open('notebooks/model.pkl','rb'))
cv = pickle.load(open('notebooks/vectorizer.pkl','rb'))


# app
app = Flask(__name__)

# routes

@app.route('/')
def hello():
    return "hello"
@app.route('/api', methods=['POST'])
def predict():
    # get data
    # return "hello"
    print("g")
    # data = request.args["text"]
    # print(request.json.get("text"))
    mg  = request.get_json()
    print(mg)

    # return jsonify(mg)
    # print(request.get_json(force=True))
    # data = json.loads(data)
    # return jsonify(results={'results':"hello"})
    
    # datag = data['text']
    

    # convert data into dataframe
    # data.update((x, [y]) for x, y in data.items())
    # data_df = pd.DataFrame.from_dict(data)
    print("gg")
    datag = mg["text"]
    
    dataa = cv.transform([datag])
    
    print("gg")
    
    

    # predictions
    result = model.predict(dataa)
    print(result[0])
    out = {"res":str(result[0])}
    return jsonify(out)

    # send back to browser
    output = {'results': int(result[0])}

    # return data
    return str(result[0])

if __name__ == '__main__':
    app.run(port = 5000, debug=True)