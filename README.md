<img src="./img/deals_laptops.jpg" alt="drawing" width="500"/>

# Prediction of laptop price with regressions
This repository stores all the code I used to win [this Kaggle competition](https://www.kaggle.com/competitions/tu-mejor-portatil-com), where the laptop price had to be predicted given their specifications.

## Main features of the project:
- Deep cleaning of the variables, looking to get as many numerical variables as possible. **Webscrapping** was used to get the GPU and CPU performance as a proxy for their price. The manufacturer and type was also given a value, by the median price of each type.
- Transformations and analysis of the models: standard scalling, polinomical features, cross-validation and grid search
- Several logarithms were tested: linear, polinomical, gradient boost and random forest. The winning model was a **gradient boost with the polinomical features of degree = 2**.

## Main files:
- [Work journal](work_journal.ipynb) is the main file for the project. Visualizations, modelization and utilities usage is found here.
- [Utils](utils.py) stores the functions used for the project.
