import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from tkinter import *

# Step 1: Load and preprocess the dataset
file_path = 'D:/desktop/content ML/Internship on learning disability detector and classifier system/disablility_detector.csv'  # Replace with the correct file path and name
df = pd.read_csv(file_path)

# Separate numeric and categorical columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Handle missing values for numeric columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Handle missing values for categorical columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# Define the features (X) and target (y)
y = df['ASD_traits']  # This is the target variable
df = df.drop(['CASE_NO_PATIENTS', 'ASD_traits'], axis=1)  # Exclude non-informative columns


categorical_cols_new = df.select_dtypes(include=['object']).columns

label_encoders = {}
categorical_columns = ['Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD', 'Who_completed_the_test']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


# Encode categorical features
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])
df['Ethnicity'] = label_encoder.fit_transform(df['Ethnicity'])
df['Jaundice'] = label_encoder.fit_transform(df['Jaundice'])
df['Family_mem_with_ASD'] = label_encoder.fit_transform(df['Family_mem_with_ASD'])
df['Who_completed_the_test'] = label_encoder.fit_transform(df['Who_completed_the_test'])
df['Speech Delay/Language Disorder'] = label_encoder.fit_transform(df['Speech Delay/Language Disorder'])
df['Learning disorder'] = label_encoder.fit_transform(df['Learning disorder'])
df['Genetic_Disorders'] = label_encoder.fit_transform(df['Genetic_Disorders'])
df['Depression'] = label_encoder.fit_transform(df['Depression'])
df['Global developmental delay/intellectual disability'] = label_encoder.fit_transform(df['Global developmental delay/intellectual disability'])
df['Social/Behavioural Issues'] = label_encoder.fit_transform(df['Social/Behavioural Issues'])
df['Anxiety_disorder'] = label_encoder.fit_transform(df['Anxiety_disorder'])



y=label_encoder.fit_transform(y)

X = pd.get_dummies(df, columns=categorical_cols_new, drop_first=True)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 2: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Step 3: Create the GUI using tkinter
def check_disability():
    # Get user input
    inputs = [
        int(a1_var.get()), int(a2_var.get()), int(a3_var.get()), int(a4_var.get()), int(a5_var.get()),
        int(a6_var.get()), int(a7_var.get()), int(a8_var.get()), int(a9_var.get()), int(a10_var.get()),
        int(social_resp_var.get()), int(age_var.get()), int(qchat_var.get()), (speech_var.get()),
        (learning_var.get()), (genetic_var.get()), (depression_var.get()),
        (global_var.get()), (social_behave_var.get()), int(autism_rating_var.get()),
        (anxiety_var.get()), sex_var.get(), ethnicity_var.get(), jaundice_var.get(),
        family_asd_var.get(), who_completed_var.get()
    ]
    
    # Create a DataFrame from inputs
    input_df = pd.DataFrame([inputs], columns=[
        'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10_Autism_Spectrum_Quotient',
        'Social_Responsiveness_Scale', 'Age_Years', 'Qchat_10_Score', 'Speech Delay/Language Disorder',
        'Learning disorder', 'Genetic_Disorders', 'Depression', 'Global developmental delay/intellectual disability',
        'Social/Behavioural Issues', 'Childhood Autism Rating Scale', 'Anxiety_disorder',
        'Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD', 'Who_completed_the_test'
    ])

    # Apply label encoding to categorical features using the dictionary of LabelEncoders
    for col in categorical_columns:
        input_df[col] = label_encoders[col].transform([input_df[col][0]])[0]
    
    # Create dummy variables for the input data
    input_df = pd.get_dummies(input_df, drop_first=True)
    
    # Ensure input_df has the same columns as the model's training data
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    # Standardize the input
    input_scaled = scaler.transform(input_df)
    
    # Predict using the trained model
    prediction = model.predict(input_scaled)
    
    # Display the result
    if prediction[0] == 0:
        result_var.set("No Learning Disability Detected")
    else:
        result_var.set("Learning Disability Detected")




# Initialize tkinter window
root = Tk()
root.title("Learning Disability Detector")

# Create input fields
a1_var = StringVar()
a2_var = StringVar()
a3_var = StringVar()
a4_var = StringVar()
a5_var = StringVar()
a6_var = StringVar()
a7_var = StringVar()
a8_var = StringVar()
a9_var = StringVar()
a10_var = StringVar()
social_resp_var = StringVar()
age_var = StringVar()
qchat_var = StringVar()
speech_var = StringVar()
learning_var = StringVar()
genetic_var = StringVar()
depression_var = StringVar()
global_var = StringVar()
social_behave_var = StringVar()
autism_rating_var = StringVar()
anxiety_var = StringVar()
sex_var = StringVar()
ethnicity_var = StringVar()
jaundice_var = StringVar()
family_asd_var = StringVar()
who_completed_var = StringVar()
result_var = StringVar()

# Arrange input fields in the GUI
Label(root, text="A1:").grid(row=0, column=0)
Entry(root, textvariable=a1_var).grid(row=0, column=1)

Label(root, text="A2:").grid(row=1, column=0)
Entry(root, textvariable=a2_var).grid(row=1, column=1)

Label(root, text="A3:").grid(row=2, column=0)
Entry(root, textvariable=a3_var).grid(row=2, column=1)

Label(root, text="A4:").grid(row=3, column=0)
Entry(root, textvariable=a4_var).grid(row=3, column=1)

Label(root, text="A5:").grid(row=4, column=0)
Entry(root, textvariable=a5_var).grid(row=4, column=1)

Label(root, text="A6:").grid(row=5, column=0)
Entry(root, textvariable=a6_var).grid(row=5, column=1)

Label(root, text="A7:").grid(row=6, column=0)
Entry(root, textvariable=a7_var).grid(row=6, column=1)

Label(root, text="A8:").grid(row=7, column=0)
Entry(root, textvariable=a8_var).grid(row=7, column=1)

Label(root, text="A9:").grid(row=8, column=0)
Entry(root, textvariable=a9_var).grid(row=8, column=1)

Label(root, text="A10_Autism_Spectrum_Quotient:").grid(row=9, column=0)
Entry(root, textvariable=a10_var).grid(row=9, column=1)

Label(root, text="Social_Responsiveness_Scale:").grid(row=10, column=0)
Entry(root, textvariable=social_resp_var).grid(row=10, column=1)

Label(root, text="Age_Years:").grid(row=11, column=0)
Entry(root, textvariable=age_var).grid(row=11, column=1)

Label(root, text="Qchat_10_Score:").grid(row=12, column=0)
Entry(root, textvariable=qchat_var).grid(row=12, column=1)

Label(root, text="Speech Delay/Language Disorder:").grid(row=13, column=0)
Entry(root, textvariable=speech_var).grid(row=13, column=1)

Label(root, text="Learning disorder:").grid(row=14, column=0)
Entry(root, textvariable=learning_var).grid(row=14, column=1)

Label(root, text="Genetic_Disorders:").grid(row=15, column=0)
Entry(root, textvariable=genetic_var).grid(row=15, column=1)

Label(root, text="Depression:").grid(row=16, column=0)
Entry(root, textvariable=depression_var).grid(row=16, column=1)

Label(root, text="Global developmental delay:").grid(row=17, column=0)
Entry(root, textvariable=global_var).grid(row=17, column=1)

Label(root, text="Social/Behavioural Issues:").grid(row=18, column=0)
Entry(root, textvariable=social_behave_var).grid(row=18, column=1)

Label(root, text="Childhood Autism Rating Scale:").grid(row=19, column=0)
Entry(root, textvariable=autism_rating_var).grid(row=19, column=1)

Label(root, text="Anxiety disorder:").grid(row=20, column=0)
Entry(root, textvariable=anxiety_var).grid(row=20, column=1)

Label(root, text="Sex (M=1, F=0):").grid(row=21, column=0)
Entry(root, textvariable=sex_var).grid(row=21, column=1)

Label(root, text="Ethnicity:").grid(row=22, column=0)
Entry(root, textvariable=ethnicity_var).grid(row=22, column=1)

Label(root, text="Jaundice (Yes=1, No=0):").grid(row=23, column=0)
Entry(root, textvariable=jaundice_var).grid(row=23, column=1)

Label(root, text="Family member with ASD (Yes=1, No=0):").grid(row=24, column=0)
Entry(root, textvariable=family_asd_var).grid(row=24, column=1)

Label(root, text="Who completed the test:").grid(row=25, column=0)
Entry(root, textvariable=who_completed_var).grid(row=25, column=1)

# Prediction button
Button(root, text="Check Disability", command=check_disability).grid(row=26, column=1)

# Result label
Label(root, textvariable=result_var, fg="red").grid(row=27, column=1)

# Run the application
root.mainloop()
