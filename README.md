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

Voici le lien de notre github de notre projet [BigDataAPp-Project](https://github.com/williamchnt/BigDataAPp-Project).

We separated our ML project workflow into different scripts

Documentation library used: 

### Random Forest

## Part III

### SHAP library

Pour la partie 3 je t'ai écrit un truc : 

```shell
This code uses the library SHAP, to build explanations for a machine learning model. The code first splits the data into a training set and a test set using the train_test_split function from the sklearn library.

Then, the TreeExplainer class is used to create an instance of an explanation object. This class can be used to explain the output of tree-based models, such as random forests and gradient boosting machines. The explainer is initialized with the trained model and training data.

The explainer(X_test, check_additivity=False) method is then called to compute SHAP values for the test data. These values represent the contribution of each feature to the output of the model for each test sample.

The code then uses the summary_plot function to visualize the explanations. The first plot shows the explanations for a particular point in the test data using the code line summary_plot(shap_values[:,:-1], X_test.iloc [0]). The second plot shows the explanations for all points of the test data at once using the code line summary_plot(shap_values, X_test, plot_type='bar'). The last plot shows a summary plot for each class of the entire data set using shap.summary_plot(shap_values, X_test, class_names=y_test)

In summary, the TreeExplainer class is used to create an explanation object that can be used to explain the output of tree-based models, and the code uses the summary_plot function to visualize the explanations for various points in the test data.
```
