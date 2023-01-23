# Applications of Big Data
## Graded Project

**Élèves** :  
William  CHENUT  
Nidhal  TEYEB  
Wassim  ZOUITENE  

**Enseignant** :  
Medina  HADJEM  



## Part I

### Code

In the first part of our project, we introduced tools and methods to facilitate collaboration between our team members. We chose to use Git over Github to manage the versions of our code and the different documents that were created. This allowed us to work on the same versions of the libraries and easily share changes between team members.

To manage the versions of the libraries, we decided to use an Anaconda environment. This environment allows us to manage the dependencies between the different libraries used in our project and easily install them on each team member's computer. This allowed us to work with stable versions of the libraries and avoid version conflicts.

As for the organization of the files, we decided to use the CookieCutter template. This template allowed us to structure our project in an efficient way and have a clear and easy to understand file system for all team members. We customized this template to our needs by creating folders and subfolders for each part of the project.

### Random Forest

Finally, for the classification model, we decided to use a random forest model. This model is one of the most complete and best suited for classification. It allows us to process the data efficiently and predict the expected results with great accuracy. We spent a lot of time optimizing the parameters of this model in order to obtain the best possible results.

In summary, we have implemented tools and methods to facilitate collaboration between our team members and effectively manage the different parts of our project. This allowed us to work efficiently and complete our project on time.

This is the github link of our project [BigDataAPp-Project](https://github.com/williamchnt/BigDataAPp-Project).

## Part II

We made 4 notebooks to explore the data and make the models by creating python functions accessible via a Flask API.
We can launch the API with the command Fask run.

http://127.0.0.1:5000/api/models/decision_tree/predict -> method to predict data, the parameters are: the set of features and the name of the model

http://127.0.0.1:5000/api/models/decision_tree/predict_test -> for prediction on the test dataset, parameters: "model_name", "row": row number of the dataset that we want to predict

http://127.0.0.1:5000/api/models/decision_tree/predict_test -> for train a model (Random forest) -> "n_estimators": 100,
     "max_depth",
     "max_features",
     "min_samples_leaf",
     "min_samples_split",
     "save_model",
     "random_state",
     "model_name" ,
     "record_model",
     "mlflow_experiment_name",
     "test_size",
     "random_state_split"

### Results


## Part III

### SHAP library


This code uses the library SHAP, to build explanations for a machine learning model. The code first splits the data into a training set and a test set using the train_test_split function from the sklearn library.
Then, the TreeExplainer class is used to create an instance of an explanation object. This class can be used to explain the output of tree-based models, such as random forests and gradient boosting machines. The explainer is initialized with the trained model and training data.
The explainer(X_test, check_additivity=False) method is then called to compute SHAP values for the test data. These values represent the contribution of each feature to the output of the model for each test sample.
The code then uses the summary_plot function to visualize the explanations. The first plot shows the explanations for a particular point in the test data using the code line summary_plot(shap_values[:,:-1], X_test.iloc [0]). The second plot shows the explanations for all points of the test data at once using the code line summary_plot(shap_values, X_test, plot_type='bar'). The last plot shows a summary plot for each class of the entire data set using shap.summary_plot(shap_values, X_test, class_names=y_test)
In summary, the TreeExplainer class is used to create an explanation object that can be used to explain the output of tree-based models, and the code uses the summary_plot function to visualize the explanations for various points in the test data.
