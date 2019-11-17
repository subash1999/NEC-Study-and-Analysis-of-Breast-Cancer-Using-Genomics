from django.shortcuts import render,redirect
from django.http import HttpResponse,JsonResponse,Http404
from django.utils.safestring import mark_safe
from .ml_code.predict_django import PredictDjango

predict = PredictDjango()

# Create your views here.
def index(request):	
	best_model = predict.getBestModelCSVDF().drop([
		predict.getBestModelCSVDF().columns[-1],  predict.getBestModelCSVDF().columns[-2]], axis=1)
	return render(request,'predict/index.html',{
		'page_title' : "Model Options",
		'best_model_columns' : best_model.columns,
		'best_model_rows' : best_model.iterrows(),
	})
def selectedModel(request,index,row_index):
	selected_csv = predict.getCSVOfIndex(index)
	prediction_options = []
	for x in range(0,len(selected_csv.index)):
		prediction_options.append(x)
	res = prediction(index,row_index)
	return render(request,'predict/selected_model.html',{
		'page_title' :  predict.getBestModelCSVDF().iloc[index]['Clf_Model'],
		'index' : index,
		'row' : predict.getBestModelCSVDF().iloc[index].to_dict(),
		'prediction_options' : prediction_options,		
		'actual_result' : res['actual'],
		'predicted_result' : res['predicted'],	
		'row_index' : row_index,	
	})

def prediction(model_index,row_index):
	predict.selectModel(model_index)
	res = predict.predictResult(row_index)
	return res
	
