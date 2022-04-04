from django.shortcuts import render
from . import ml_model as tm


def home(request):
    return render(request, 'index.html')


def result(request):
    pclass = request.GET['pclass']
    sex = request.GET['sex']
    age = request.GET['age']
    sibsp = request.GET['sibsp']
    parch = request.GET['pclass']
    fare = request.GET['pclass']
    embarked = request.GET['pclass']
    title = request.GET['pclass']
    prediction = tm.prediction_model(
        pclass, sex, age, sibsp, parch, fare, embarked, title)
    return render(request, 'result.html', {'Prediction': prediction})
