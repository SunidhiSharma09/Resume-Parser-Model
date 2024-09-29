#!/usr/bin/env python
# coding: utf-8

# #### Resume Parser Model

# In[ ]:





# Python script to extract text from PDF and DOCX resume files and to store the extracted data in the corresponding columns in an Excel file.

# In[2]:


# importing libraries
import os
import re
import docx
import pdfplumber  # instead of PyPDF2 for better PDF text extraction
import pandas as pd

# Helper function to extract text from docx
def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return '\n'.join([para.text for para in doc.paragraphs])

# Helper function to extract text from pdf 
def extract_text_from_pdf(file_path):
    text = ''
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to normalize text 
def normalize_text(text):
    return text.replace('\n', '').replace('\r', '').strip()

# Function to extract name (assumed to be the first string of characters in the resume)
def extract_name(text):
    name = text.split('\n')[0].strip()
    return name

# Function to extract mobile number 
def extract_mobile_number(text):
    phone_patterns = [
        re.compile(r'\+?\d{1,4}[\s.-]?\(?\d{1,4}\)?[\s.-]?\d{1,4}[\s.-]?\d{1,9}', re.IGNORECASE),
        re.compile(r'\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}', re.IGNORECASE),
        re.compile(r'\d{10}', re.IGNORECASE)
    ]
    for pattern in phone_patterns:
        match = pattern.search(text)
        if match:
            return match.group().strip()
    return None

# Fuction to extract email
def extract_email(text):
    text = normalize_text(text)
    email_regex = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', re.IGNORECASE)
    matches = email_regex.findall(text)
    return matches[0].strip() if matches else None  

# Function to extract city and state 
def extract_location(text):
    location_patterns = [
        re.compile(r'([A-Za-z\s]+),\s?([A-Za-z\s]+)', re.IGNORECASE),
        re.compile(r'([A-Za-z\s]+)\s*-\s*([A-Za-z\s]+)', re.IGNORECASE),
        re.compile(r'([A-Za-z\s]+)\s+([A-Za-z\s]+)', re.IGNORECASE)
    ]
    for pattern in location_patterns:
        match = pattern.search(text)
        if match:
            city, state = match.groups()
            return city.strip(), state.strip()
    return None, None

# Function to extract education details
def extract_education(text):
    education_keywords = ['Bachelor', 'Master', 'PhD', 'B.Sc', 'M.Sc', 'B.A', 'M.A', 'degree', 'university', 'college']
    education_section = ''
    for line in text.split('\n'):
        if any(keyword.lower() in line.lower() for keyword in education_keywords):
            education_section += line.strip() + '; '
    return education_section.strip() if education_section else None

# Function to extract skills
def extract_skills(text):
    skills_keywords = ['Skills', 'Technical Skills', 'Core Competencies', 'Proficiencies', 'Expertise']
    skill_text = ''
    
    for keyword in skills_keywords:
        keyword_regex = re.compile(f'{keyword}.*?:\n?(.+)', re.IGNORECASE | re.DOTALL)
        match = keyword_regex.search(text)
        if match:
            skill_text = match.group(1).strip().replace('\n', ', ')
            break

    if skill_text:
        unwanted_phrases = ['good at', 'familiar with', 'proficient in', 'knowledge of']
        for phrase in unwanted_phrases:
            skill_text = re.sub(phrase, '', skill_text, flags=re.IGNORECASE)
        
        skill_text = ', '.join([skill.strip() for skill in skill_text.split(',') if skill.strip()])
    
    return skill_text if skill_text else None

# Function to extract experience and 0 if no experience found
def extract_experience(text):
    experience_patterns = [
        re.compile(r'(\d+)\s+years? of experience', re.IGNORECASE),
        re.compile(r'experience of\s?(\d+)\s+years?', re.IGNORECASE),
        re.compile(r'(\d+)\s+years? experience', re.IGNORECASE),
        re.compile(r'experience\s*:\s*(\d+)\s+years?', re.IGNORECASE),
        re.compile(r'(\d+)[+\s]*years experience', re.IGNORECASE)
    ]
    
    for pattern in experience_patterns:
        match = pattern.search(text)
        if match:
            return match.group(1)  

    return '0'  

# Function to extract primary skill
def extract_primary_skill(text):
    primary_skill = {
        'Data Analyst': ['data analyst'],
        'Data Scientist': ['data scientist'],
        'Data Engineer': ['data engineer'],
        'Cloud Engineer': ['cloud engineer'],
        'Web Developer': ['web developer', 'frontend', 'backend'],
        'Digital Marketing': ['digital marketing', 'seo', 'social media'],
        'Software Developer': ['software developer', 'programmer'],
        'Software Engineer': ['software engineer']
    }
    
    for primary_skill, keywords in primary_skill.items():
        for keyword in keywords:
            if re.search(keyword, text, re.IGNORECASE):
                return primary_skill
    
    return 'Unknown'  

# Function to clean text and remove illegal characters
def clean_text(text):
    if text is None:
        return ''
    return re.sub(r'[^\x20-\x7E]', '', text)  # Remove non-printable characters

# Function to process each resume
def process_resume(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.docx':
        text = extract_text_from_docx(file_path)
    elif ext == '.pdf':
        text = extract_text_from_pdf(file_path)  
    else:
        return None  
    
    name = clean_text(extract_name(text))
    mobile = clean_text(extract_mobile_number(text))
    email = clean_text(extract_email(text))  
    city, state = (clean_text(loc) for loc in extract_location(text))
    education = clean_text(extract_education(text))
    skills = clean_text(extract_skills(text))
    experience = clean_text(extract_experience(text))
    primary_skill = extract_primary_skill(text)  

    return {
        'Name': name,
        'Mobile': mobile,
        'Email': email,
        'City': city,
        'State': state,
        'Education': education,
        'Skills': skills,
        'Experience (Years)': experience,
        'Primary_Skill': primary_skill  
    }

# Function to process all resumes in the folder
def process_all_resumes(folder_path):
    resume_data = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_path.endswith(('.docx', '.pdf')):
            data = process_resume(file_path)
            if data:
                resume_data.append(data)

    return resume_data

# Function to save extracted resume data to an Excel file
def save_to_excel(resume_data, output_file):
    df = pd.DataFrame(resume_data)
    df.to_excel(output_file, index=False)
    print(f"Data saved to {output_file}")

folder_path = r'C:\Users\DELL\Documents\RESUMES'  # Folder containing the resumes
output_file = r'C:\Users\DELL\Documents\Resume_data.xlsx' # excel filr to save extracted data

# Process resumes and save to Excel
resume_data = process_all_resumes(folder_path)
save_to_excel(resume_data, output_file)


# In[ ]:





# Python script that uses the XGBoost algorithm to predict the primary skill/Job Role of a resume based on its content.

# In[45]:


# Checking class distribution by importing cleaned resume data for developing XGBoost model.

import pandas as pd

resume_data = pd.read_excel(r'C:\Users\DELL\Documents\Resume_data.xlsm', 'Cleaned_resume_data')

class_distribution = resume_data['Primary_Skill'].value_counts()
print("Class Distribution (Number of samples in each class):\n")
print(class_distribution)


# In[ ]:


# This is imbalanced data so we need to do upsampling before creating a model.


# In[39]:


# Developing XGBoost model for primary skill prediction based on resume data.

# Importing libraries.
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

class ResumeSkillPredictor:  # creating class for prediction.
    def __init__(self, filepath, sheet_name):
        self.filepath = filepath
        self.sheet_name = sheet_name
        self.model = None
        self.label_encoder = LabelEncoder()

    def load_and_preprocess_data(self): # function to load and pre-process data.
        resume_data = pd.read_excel(self.filepath, self.sheet_name)

        # Separating Data Analyst data from other classes
        data_analyst_data = resume_data[resume_data['Primary_Skill'] == 'Data Analyst']
        other_data = resume_data[resume_data['Primary_Skill'] != 'Data Analyst']

        # Upsampling other classes to match the size of the Data Analyst class
        upsampled_data = pd.DataFrame()
        for skill in other_data['Primary_Skill'].unique():
            skill_data = other_data[other_data['Primary_Skill'] == skill]
            upsampled_skill_data = resample(
                skill_data,
                replace=True,
                n_samples=len(data_analyst_data),
                random_state=0
            )
            upsampled_data = pd.concat([upsampled_data, upsampled_skill_data])
        # Combining upsampled data with Data Analyst data
        combined_data = pd.concat([data_analyst_data, upsampled_data])

        # Shuffling the dataset
        combined_data = combined_data.sample(frac=1, random_state=0).reset_index(drop=True)

        # separating features and target into X and y.
        X = combined_data.drop(['Primary_Skill', 'Name', 'Mobile', 'Email', 'City', 'State', 'Education'], axis=1, errors='ignore')
        y = combined_data['Primary_Skill']

        # One-hot encoding the skills in 'Skills' column
        X = pd.get_dummies(X, columns=['Skills'], drop_first=True)

        # Label encoding for the target column (Primary_Skill)
        y = self.label_encoder.fit_transform(y)

        # Splitting the data into training and testing sets.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):  # function to train the model using GridSearchCV
        xgb_model = xgb.XGBClassifier(eval_metric='mlogloss')  # defining XGBoost model

        # Defining parameter grid for GridSearchCV
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        # Applying GridSearchCV with 5-fold cross-validation
        grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Getting the best model
        self.model = grid_search.best_estimator_

        # Evaluating the model using cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.2f}")

    def evaluate_model(self, X_test, y_test):    # function to evaluate the model
        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        # Generating a classification report
        report = classification_report(
            y_test, y_pred,
            labels=list(range(len(self.label_encoder.classes_))),
            target_names=self.label_encoder.classes_,
            zero_division=0
        )

        print(f"Model Accuracy: {accuracy:.2f}")
        print("\nClassification Report:\n", report)

        # Generating Confusion Matrix table with all labels
        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_matrix_df = pd.DataFrame(
            conf_matrix,
            index=[f"Actual: {label}" for label in self.label_encoder.classes_],
            columns=[f"Predicted: {label}" for label in self.label_encoder.classes_]
        )
        print("\nConfusion Matrix with Labels:\n", conf_matrix_df)

        # Confusion Matrix Visualization (Heatmap)
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix Heatmap')
        plt.xlabel('Predicted Labels')
        plt.ylabel('Actual Labels')

        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        plt.show()

    def predict_primary_skill(self, new_resume_data):   # function to make prediction for new resume data
        new_resume_data = pd.get_dummies(new_resume_data, columns=['Skills'], drop_first=True)
        return self.model.predict(new_resume_data)

# path of the cleaned extracted resume data.
filepath = r'C:\Users\DELL\Documents\Resume_data.xlsm'
sheet_name = 'Cleaned_resume_data'

# Output Generation by calling class and different functions.
skill_predictor = ResumeSkillPredictor(filepath, sheet_name)

X_train, X_test, y_train, y_test = skill_predictor.load_and_preprocess_data()

skill_predictor.train_model(X_train, y_train)

skill_predictor.evaluate_model(X_test, y_test)


# In[ ]:





# Statistical analysis of the resume data as well as visualization of certain data.

# In[1]:


# Visualization of the resume data analysis and statistical analysis of the numerical column.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class ResumeAnalyzer:
    def __init__(self, resume_data):
        self.resume_data = resume_data
    
    def visualize_experience_distribution(self): # function to visualize resumes/applicants count along the years of experience
        plt.figure(figsize=(10, 6))
        ax = sns.histplot(self.resume_data['Experience (Years)'].dropna(), bins=10, kde=True)
        plt.title('Distribution of Years of Experience')
        plt.xlabel('Years of Experience')
        plt.ylabel('Frequency')

        # to add data labels
        for p in ax.patches:
            height = p.get_height()
            if height > 0:  # Avoid adding labels on empty bars
                ax.text(p.get_x() + p.get_width() / 2., height + 0.2, int(height), ha="center")
        
        plt.show()
    
    def visualize_skill_frequency(self):  # function to visualize the number of applicants/resumes based on different skills.
        skills_series = self.resume_data['Skills'].dropna().str.split(',').explode().str.strip()  # separating each comma separated skill
        skill_counts = skills_series.value_counts()
        
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x=skill_counts.values, y=skill_counts.index)
        plt.title('Frequency of Different Skills')
        plt.xlabel('Frequency')
        plt.ylabel('Skills')
        
        # to add data labels
        for i, v in enumerate(skill_counts.values):
            ax.text(v + 0.5, i, str(v), color='black', va='center')
        
        plt.show()
    
    def visualize_primary_skill_frequency(self):  # function to visualize the number of resumes based on each primary skill.
        primary_skill_counts = self.resume_data['Primary_Skill'].dropna().value_counts()
        
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x=primary_skill_counts.index, y=primary_skill_counts.values)
        plt.title('Frequency of Primary Skills')
        plt.xlabel('Primary Skill')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        
        # to add data labels
        for i, v in enumerate(primary_skill_counts.values):
            ax.text(i, v + 0.5, str(v), ha='center', va='bottom')
        
        plt.show()
    
    def get_statistical_summary(self):  # function to generate statistical summary
        summary = self.resume_data.describe(include=[np.number])
        return summary

# File path to load data
file_path = 'C:\\Users\\DELL\\Documents\\Resume_data.xlsm'
sheet_name = 'Cleaned_resume_data'
resume_data = pd.read_excel(file_path, sheet_name=sheet_name)

# Generating output by call class and different functions.
analyzer = ResumeAnalyzer(resume_data)
analyzer.visualize_experience_distribution()
analyzer.visualize_skill_frequency()
analyzer.visualize_primary_skill_frequency()
print('Statistical Summary\n', analyzer.get_statistical_summary())


# In[ ]:





# In[ ]:




