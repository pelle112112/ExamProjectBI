from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import readData
from sklearn.metrics.pairwise import linear_kernel

# Reading data, saved as pandas dataframe
data = readData.loadData('../Data/exercise_dataset.csv', 'csv')

# No null values
print(data.isnull().sum())

# Creating tfdif matrix for features we want similarities for - in this case we combine what is in the 'Activity, Exercise or Sport (1 hour)' feature and the 'Calories per kg'. 
tfidf = TfidfVectorizer(stop_words='english')
data['featuresForTraining'] = data['Activity, Exercise or Sport (1 hour)'] + ' ' + data['Calories per kg'].astype(str)
tfidf_matrix = tfidf.fit_transform(data['featuresForTraining'])

print(tfidf_matrix.shape)
# We use cosine score as a metric for similarity, and calculates cosine score per data entry. 
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

print(cosine_sim)

# Creating a reverse map using pandas, where the key is the name of the activity and the value is the index number of the activity. 
indices = pd.Series(data.index, index=data['featuresForTraining'])

# This function takes an exercise entry and returns a list of recommended exercises which have content similarities to the entered exercise. 
def get_recommendations(exercise):
    # Find the index of the exercise we want to find similar exercises for. 
    idx = indices[exercise]
    # Get a pairwise similarity score of all exercises with the exercise we found the index for in the step above.
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the pairs, so we get the pairs with the highest cosine score first.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the pairs of indexes and scores of the 19 most similar exercises. We want a maximum of ten but because we don't want the same kind of activities to many times, we take a few extra, which if needed will be filtered out later.
    sim_scores = sim_scores[1:20]
    # List of similar exercises and dictionary of which activities already has been chosen as a recommended activity.
    recommendedExercises = []
    chosenActivities = {}

    # Adding the activity of the exercise we wanted to search for similarities for, to the dictionary with a count of 1. 
    exerciseActivity = data['Activity, Exercise or Sport (1 hour)'].iloc[idx].split(',')[0]
    chosenActivities[exerciseActivity] = 1

    #Looping through the pairs of indexes and cosine scores, to determine which should be added. 
    for i, score in sim_scores:
        # Finding the general activity for example cycling, aerobics etc.
        activity = data['Activity, Exercise or Sport (1 hour)'].iloc[i]
        activityName = activity.split(',')[0]
        
        # We only want to recommend an exercise if it is somehow similar to the original exercise, therefore if the cosine score is 0, we do not care about the exercise. 
        if score is not 0:
            # If the name of the activity already exists in the dictionary we use the dictionary to make sure that there isn't a lot of the same entries in the recommendations. For example 10 recommendations of running at different speeds.
            if activityName in chosenActivities:
                if chosenActivities[activityName] < 2:
                    recommendedExercises.append(activity)
                    # If the exercise is added to the recommendations, the counter of the dictionary is counted upwards, to keep track of how many variations of the exercise is in the recommendations. 
                    chosenActivities[activityName] += 1
            else: 
                recommendedExercises.append(activity)
                # If the exercise was not already in the dictionary it is added with an initial count of 1. 
                chosenActivities[activityName] = 1

        # In case we get more than 10 valid recommendations, we stop looking for more when the limit has been reached.
        if len(recommendedExercises) == 10:
            return recommendedExercises

    return recommendedExercises

print(get_recommendations(data['Activity, Exercise or Sport (1 hour)'][0] + ' ' + data['Calories per kg'][0].astype(str)))