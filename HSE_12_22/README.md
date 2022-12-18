# Hackathon from the Higher School of Economics: "Airline passenger satisfaction"

## Task part №1 content

1.Check for missing values.

2.Carry out univariate analysis. Use .describe(), vizualization and other methods to check out the distribution of the columns. Are there any 
outliers? If there are, you can drop them or replace them similarly to missing values. There are also a bunch of other methods to work with 
outliers, feel free to do more research!

3.Carry out multivariate analysis. For example, you can use scatter plots and a correlation matrix. Side note: keep in mind that correlation 
only checks for linear dependencies. If the correlation is small, it doesn't mean that there is no dependency at all, only that there is no linear 
dependency.

4.Use grouping (.group), filterings (for example, like this df[df[col] > df[col].quantile(.95)]), vizualizations to formulate different 
hypothesis 
about the data. For example, maybe loyal customers are usually business travelers? Check it out! Don't forget to write down your conclusions.

## Task part №2 content

1.Create new features based on your EDA. Don't forget to check how they performed after you are finished with modelling! You can use 
feature_importances from scikit-learn or use advanced methods like SHAP or Lime.

2.Your target variable is satisfaction. You should research metrics and choose one or multiple that you will use to validate your model. Write 
down the formula(s) and your motivation to use them.

3.Design the validation process: for example, will you use cross-validation or just train-test split? Will you account for the imbalance in 
classes, if it exists?

4.Experiment with modelling. You can use models from the lecture or do your own research. You can also try out approaches like stacking and 
blending – will they increase the quality?
Make predictions on the test.csv dataset.

## RESULT

- The team took the second place.(Accuracy: 0.9642)

## Teammates:

[Andrey Baranov](https://github.com/and9331) 
