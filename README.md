## RESUME PARSER MODEL
- Live dashboard:[ https://app.powerbi.com/reportEmbed?reportId=50962eaa-18b3-44fe-a200-357b38557b18&autoAuth=true&ctid=f32b2380-e473-4691-8ba9-71915e0a20cd](https://app.powerbi.com/view?r=eyJrIjoiNzY5MDhhMWQtNDkyNC00YWRmLWFlYjAtNTAyYWUwY2Q3MDE0IiwidCI6ImYzMmIyMzgwLWU0NzMtNDY5MS04YmE5LTcxOTE1ZTBhMjBjZCJ9&pageName=d690843bd775ad42617d)
### Introduction
In today's fast-paced job market, companies receive a large number of resumes for every open position, making it time-consuming and labor-intensive for HR professionals to manually review 
and assess each resume. Extracting key information such as candidate details, skills, experience etc. from diverse resume formats can be cumbersome and prone to errors.
To streamline this process, the Resume Parser project aims to automate resume data extraction, processing, and analysis.This automation not only saves time but also improves the accuracy and 
efficiency of candidate assessment, and helps organizations make data-driven decisions while shortlisting and interviewing candidates.
### Objective
To develop an automated system for extracting, cleaning, storing, and analyzing resume data using Python, SQL, machine learning, and data visualization tools. Ensure efficient 
data handling, robust analysis, and insightful visualizations to streamline the management of resume information and enhance data-driven decision-making
### Tools & Techniques Implemented
- Python : It is used to extract key information from the resumes and store it in the excel file.
- Excel : It is used to clean the extracted data using Power Query and a VBA automation is also developed to standardised the data for analysis. Further a VBA Automation is developed to automatically store the new records in the SQL Server Database as sson as it is entered in the excel sheet.
- SQL Server : The cleaned data is then stored in the database in SQL Server from excel using Stored Procedure during which a unique Resume ID is given to each resume. A trigger is also created to log the insertion of every resume with date, time and a unique log ID. Further analysis is done and various views are also created which were used during data visualization too.
- Power BI : A report containing two dasboards, one for overview of ll the resumes and other to comapre resumes, was developed. The data for this report is connected live to SQL Server database via Direct Query Mode so that any update of data in database gets directly reflected to the dashboard, which will make it easier to analyze resumes and make decisions.
- Machine Learning Algorithm : A Machine learning model to predict the Primary Skill of the candidates based on the resume content is developed using XGBoost aalgorithm. The model's performance is also presented via Classification report and Confusion matrix. This model will further improve the decision to select the right role for the candidate.
### Future Enhancement
Future enhancements could include integrating real-time resume submission through web forms, extracting information from unstructured and multi-lingual resume, and refining the machine learning models for better skill prediction.

### Note: You can find the workflow, codes screenshot and various outputs in the documentation too.
