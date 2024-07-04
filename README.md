# churn_modeling
<h1 id= "top doc"> Description </h1>

---




1. General info

Customer churn prediction is to measure why customers are leaving a business. In this tutorial we will be looking at customer churn in telecom business. We will build some models to predict the churn and use precision,recall, f1-score to measure performance of our model.   

2. Used methods

we used ***Grid search*** to find the best parameters of the most famous **XGBoost** algorithm, we also used the following methods and libraries

- Pandas

- Data visualization

- Classification

And also we tackled the ***imbalanced dataset*** throughout our code

# Highlights



 ```python

def parameter_finder (model, parameters):
    
   
    
    grid = GridSearchCV(model, 
                        param_grid = parameters, 
                        refit = True, 
                        cv = KFold(shuffle = True, random_state = 1), 
                        n_jobs = -1)
    grid_fit = grid.fit(X_train, y_train)
    y_train_pred = grid_fit.predict(X_train)
    y_pred = grid_fit.predict(X_test)
    
    train_score =grid_fit.score(X_train, y_train)
    test_score = grid_fit.score(X_test, y_test)
    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
    
    model_name = str(model).split('(')[0]
    
   
    
    print(f"The best parameters for {model_name} model is: {grid_fit.best_params_}")
    print("--" * 10)
    print(f"(R2 score) in the training set is {train_score:0.2%} for {model_name} model.")
    print(f"(R2 score) in the testing set is {test_score:0.2%} for {model_name} model.")
    print(f"RMSE is {RMSE:,} for {model_name} model.")
    print("--" * 10)
    
    
       
    return train_score, test_score, RMSE

 ```





- link to some where <a href= "#top doc"> go to up </a>
