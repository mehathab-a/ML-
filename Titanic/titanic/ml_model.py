def prediction_model(pclass, sex, age, sibsp, parch, fare, embarked, title):
    import pickle
    x = [[pclass, sex, age, sibsp, parch, fare, embarked, title]]
    randomforest = pickle.load(open('titanic_mode.sav', 'rb'))
    predictions = randomforest.predict(x)
    predictions = "Survived" if predictions == 1 else "Not Survived"
    return predictions
