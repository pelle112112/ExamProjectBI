# ExamProjectBI

### Implementation instructions:
1. Download or clone the project
2. Run "pip install -r requirements.txt" in the root of the project
3. run "streamlit run app.py"

### Problem Statement:
 How can BI and AI be leveraged to develop a personalized fitness training program for individuals, 
 and give an estimate of the possible weight loss?
1. How does an individual determine the amount of weight they will lose based on their training schedule?
2. In the absence of personal trainers and fitness expertise, 
what methods can individuals utilize to develop an effective training program?
3. What statistical data sources exist to evaluate the efficacy of different exercises for weight loss?
4. How can a personalized training program account for individual differences in intensity preferences and time availability?
5. What role does goal-setting play in fitness training, and how can individuals determine the achievability of their goals?

### Motivation:
Weight loss is difficult for many people around the world, and as the world population becomes heavier and heavier, 
we want to address the problem with a program that can predict the amount of weight individuals can have with certain training hours, 
intensity, starting weight etc, making it easier for people to set realistic goals and determine what kind of training is best.

### Theoretical Foundation:
We are using Python as the programming language and anaconda as interpreter.
We have used several models for training and predicting data.
1. linear, multilinear, polynomiel regression, random forest classifier, Naive Bayes (supervised machine learning)
2. k-Means, HierarchicalClustering (un-supervised training)

### Argumentation of Choices:
For the supervised training we started by testing which of the models gave the best accuracy. For Regression it was the R-Squared score.
And used the one with the highest score. The score is a percentage, the Higher the percentage the better.
We concluded that classification was not the best way for predicting weight loss, as it defines classes and not numerical values.

For the un-supervised training we based the choise from the silhouette score.
the score varies from (-1 to 1) the closer we get to 1 the better is the score.
### Design:
We decided to split the different aspects of the project into different folders. 
#### eksamples: Code, Data, Documentation/Graphs, Media, Model, pages.
#### Code:
All the python scripts are in this folder.
#### Data:
All the data we used are in here.
#### Documentation:
Is the visual part of the project of the different models such as graphs, plots etc.
But olso kind of our backlog since we have (Brainstorm.md, Problem-Statement.md and Problem-StatementUpdated.md).
### Code:

### Artifacts:

### Outcomes:

