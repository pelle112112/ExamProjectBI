from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import readData
from sklearn.metrics.pairwise import linear_kernel

data = readData.loadData('../Data/exercise_dataset.csv', 'csv')

print(data.isnull().sum())

tfidf = TfidfVectorizer(stop_words='english')
data['featuresForTraining'] = data['Activity, Exercise or Sport (1 hour)'] + ' ' + data['Calories per kg'].astype(str)
tfidf_matrix = tfidf.fit_transform(data['featuresForTraining'])

print(tfidf_matrix.shape)

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

print(cosine_sim)

indices = pd.Series(data.index, index=data['featuresForTraining'])

def get_recommendations(exercise):
    idx = indices[exercise]

    sim_scores = list(enumerate(cosine_sim[idx]))
    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    sim_scores = sim_scores[1:20]
    
    recommendedExercises = []
    chosenActivities = {}

    exerciseActivity = data['Activity, Exercise or Sport (1 hour)'].iloc[idx].split(',')[0]
    chosenActivities[exerciseActivity] = 1
    for i, score in sim_scores:
        activity = data['Activity, Exercise or Sport (1 hour)'].iloc[i]
        activityName = activity.split(',')[0]
        
        if score is not 0:
            if activityName in chosenActivities:
                if chosenActivities[activityName] < 2:
                    recommendedExercises.append(activity)
                    chosenActivities[activityName] += 1
            else: 
                recommendedExercises.append(activity)
                chosenActivities[activityName] = 1

        if len(recommendedExercises) == 10:
            return recommendedExercises

    return recommendedExercises

print(get_recommendations(data['Activity, Exercise or Sport (1 hour)'][0] + ' ' + data['Calories per kg'][0].astype(str)))