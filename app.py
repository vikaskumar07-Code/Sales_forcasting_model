from flask import Flask,request,render_template
from src.pipeline.predict_pipeline import PredictPipeline,CustomData
app=Flask(__name__)

#Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            Store=int(request.form["Store"]),
            Holiday_Flag=int(request.form["Holiday_Flag"]),
            Fuel_Price=float(request.form["Fuel_Price"]),
            CPI=float(request.form["CPI"]),
            Unemployment=float(request.form["Unemployment"]),
            Temperature_C=float(request.form["Temperature_C"]),
            Year=int(request.form["Year"]),
            Month_Name=request.form["Month_Name"]
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)

        return render_template('home.html',results=results[0])
    
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)
