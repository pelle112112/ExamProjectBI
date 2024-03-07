import pandas as pd
import numpy as np

# Define the number of rows in the dataset
num_rows = 1500

# Generate random data for each column
np.random.seed(0)  # for reproducibility

# Generate Gender data (50% chance of being male or female)
gender = np.random.choice(['Male', 'Female'], size=num_rows)

# Generate Starting_Weight data (random values between 60kg and 120kg dependent on gender)
starting_weight = []
for sex in gender: 
    if(sex == 'Male'):
        starting_weight.append(np.random.uniform(80, 120))
    else:
        starting_weight.append(np.random.uniform(60, 120))    

# Generate Duration_in_weeks data (random values between 6 and 20 weeks)
duration_in_weeks = np.random.randint(6, 21, size=num_rows)

# Generate Training_hours_per_week data (random values between 2 and 12 hours)
training_hours_per_week = np.random.randint(2, 13, size=num_rows)

# Generate Intensity data (randomly assigned from 'High', 'Medium', 'Low')
intensity = np.random.choice(['High', 'Medium', 'Low'], size=num_rows)

# Calculate End_Weight based on Duration_in_weeks, Training_hours_per_week, and Intensity
def calculate_end_weight(duration, hours_per_week, intensity, starting_weight):

    # 7000-8000 calories to lose 1kg fat. 

    if intensity == 'High':
        loss_factor = (starting_weight * 4.343 * hours_per_week) / np.random.uniform(7000, 8000)  
        # 1.974 calories pr. pound. * 2.2 = 4.343 pr. kg.
    elif intensity == 'Medium':
        loss_factor = (starting_weight * 3.227 * hours_per_week) / np.random.uniform(7000, 8000)  
        # 1.467 calories pr. kg. * 2.2 = 3.227 pr. kg. 
    else:
        loss_factor = (starting_weight * 2.002 * hours_per_week) / np.random.uniform(7000, 8000)  
        # 0.91 calories pr .kg. *2.2 = 2.002 pr. kg. 
    total_loss = loss_factor * duration
    end_weight = starting_weight - total_loss
    return end_weight

# Calculate End_Weight for each row
end_weight = [calculate_end_weight(duration, hours, intensity, start_weight) for duration, hours, intensity, start_weight in zip(duration_in_weeks, training_hours_per_week, intensity, starting_weight)]

# Create a DataFrame with the generated data
data = pd.DataFrame({
    'Gender': gender,
    'Starting_Weight': starting_weight,
    'End_Weight': end_weight,
    'Duration_in_weeks': duration_in_weeks,
    'Training_hours_per_week': training_hours_per_week,
    'Intensity': intensity
})

# Save the DataFrame to a CSV file
data.to_csv('weight_loss_dataset.csv', index=False)