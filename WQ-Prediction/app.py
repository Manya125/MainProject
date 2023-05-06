from flask import Flask,render_template,request
import pickle
import numpy as np

model2 = pickle.load(open('mdl.pkl','rb'))

app = Flask(__name__,template_folder="template")

@app.route('/')
def man():
    return render_template("index.html")

@app.route('/check',methods=['POST'])
def home():
    data1 = request.form['t1']
    data2 = request.form['t2']
    data3 = request.form['t3']
    data4 = request.form['t4']
    data5 = request.form['t5']
    data6 = request.form['t6']
    data7 = request.form['t7']
    data8 = request.form['t8']
    data9 = request.form['t9']
    data10 = request.form['t10']
    data11 = request.form['t11']
    data12 = request.form['t12']
    data13 = request.form['t13']
    data14 = request.form['t14']
    data15 = request.form['t15']
    data16 = request.form['t16']
    data17 = request.form['t17']
    data18 = request.form['t18']
    data19 = request.form['t19']
    data20 = request.form['t20']
    arr = np.array([[data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,data14,data15,data16,data17,data18,data19,data20]])
    pred = model2.predict(arr)
    return render_template("result.html",data=pred)

if __name__ == "__main__":
    app.run(debug=True)