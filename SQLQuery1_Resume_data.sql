/* END CAPSTONE-1: RESUME PARSER */

CREATE DATABASE RESUME;

USE RESUME;

SELECT * FROM Staging_Resume_data; -- to see the resume data stored temporarily in Staging_Resume_data table.

/* Creating Final table to store resume data  after execution of stored procedure along with assigning resumeid. Logid will be assigned
on execution of trigger. This final table will be then used for further analysis. */ 
CREATE TABLE Resume_data (
    ResumeID INT IDENTITY(1,1) PRIMARY KEY,
    Name NVARCHAR(100),
    Mobile NVARCHAR(20),
    Email NVARCHAR(100),
    City NVARCHAR(50),
    State NVARCHAR(50),
    Education NVARCHAR(100),
    Skills NVARCHAR(MAX),
    Experience_Years NVARCHAR(10),
    Primary_Skill NVARCHAR(50)
);


SELECT * FROM Resume_data;

/* Create a SQL trigger that automatically logs the insertion of newresume data into a separate log table. The log should include 
thetimestamp of the insertion and the ID of the new resume entry. */

-- creating resume insertion log table 
CREATE TABLE Resume_Insertion_Log (
    LogID INT IDENTITY(1,1) PRIMARY KEY,
    ResumeID INT,
    Insertion_Time_stamp DATETIME DEFAULT GETDATE()
);

SELECT * FROM Resume_Insertion_Log 

-- Creating Trigger
GO
CREATE TRIGGER Log_Resume_Insertion
ON Resume_data
AFTER INSERT
AS
BEGIN
    INSERT INTO Resume_Insertion_Log (ResumeID)
    SELECT ResumeID
    FROM inserted;
END;

/* Develop a stored procedure in SQL that accepts multiple resume records and inserts them into the database in a single call.
The procedure should ensure that either all records are inserted successfully, or none are, using transaction control.*/

CREATE TYPE ResumeDataType AS TABLE (
    ResumeID INT IDENTITY(1,1) PRIMARY KEY,
    Name NVARCHAR(100),
    Mobile NVARCHAR(20),
    Email NVARCHAR(100),
    City NVARCHAR(50),
    State NVARCHAR(50),
    Education NVARCHAR(100),
    Skills NVARCHAR(MAX),
    Experience_Years NVARCHAR(10),
    Primary_Skill NVARCHAR(50)
);


-- Stored Procedure for batch resume 
GO
CREATE PROCEDURE Insert_Batch_Resumes
    @ResumeData ResumeDataType READONLY -- Table-valued parameter
AS
BEGIN
    BEGIN TRANSACTION;
    BEGIN TRY
        INSERT INTO Resume_data(Name, Mobile, Email, City, State, Education, Skills, Experience_Years, Primary_Skill)
        SELECT Name, Mobile, Email, City, State, Education, Skills, Experience_Years, Primary_Skill
        FROM @ResumeData;
        COMMIT TRANSACTION;
    END TRY
    BEGIN CATCH
        ROLLBACK TRANSACTION;
    END CATCH;
END;

-- execution

DECLARE @ResumeData ResumeDataType;

INSERT INTO @ResumeData
SELECT  Name, Mobile, Email, City, State, Education, Skills, Experience_Years, Primary_Skill
FROM Staging_Resume_data;

EXEC Insert_Batch_Resumes @ResumeData;

SELECT * FROM Resume_Insertion_Log -- to check insertion log


/* Write a SQL transaction that inserts new resume data into the database. Ensure the transaction includes a check that either 
all  changes  are  committed  or  none  are,  to  maintain  data integrity. */

-- Stored Procedure for single resume insertion
CREATE PROCEDURE Insert_Single_Resume
   @Name NVARCHAR(100),
   @Mobile NVARCHAR(20),
   @Email NVARCHAR(100),
   @City NVARCHAR(50),
   @State NVARCHAR(50),
   @Education NVARCHAR(100),
   @Skills NVARCHAR(MAX),
   @Experience_Years NVARCHAR(10),
   @Primary_Skill NVARCHAR(50)
AS
BEGIN
    BEGIN TRANSACTION;
    BEGIN TRY
        INSERT INTO Resume_data(Name, Mobile, Email, City, State, Education, Skills, Experience_Years, Primary_Skill)
        VALUES (@Name,@Mobile, @Email, @City, @State, @Education, @Skills, @Experience_Years, @Primary_Skill);
        COMMIT TRANSACTION;
    END TRY
    BEGIN CATCH
        ROLLBACK TRANSACTION;
    END CATCH;
END;

-- execution
EXEC Insert_Single_Resume @Name='UMAR ALI',@Mobile='883-992-7221', @Email='ali@gmail.com', @City='Hyderabad', @State='Telangana', @Education='Bachelor of Engineering', @Skills='SQL, Power BI, React', @Experience_Years='1', @Primary_Skill='Data Analyst';

SELECT * FROM Resume_Insertion_Log -- to check insertion log

-------------------------------------------------------------------------------------------------------------------------------------------
/* Creating Views to be used in data visualization in Power BI*/

-- View for list of all the skills in the resume and the count applicants having those skills.

CREATE VIEW Skills_list AS
WITH Skills_CTE AS (
    SELECT       -- Cleaning the skills extracted for analysis
        ResumeID, 
        LTRIM(RTRIM(REPLACE(Skill.value, 'non-', ''))) AS Raw_Skill 
    FROM 
        Resume_data
    CROSS APPLY STRING_SPLIT(Skills, ',') AS Skill 
),
Split_Skills_CTE AS (  
    SELECT 
        ResumeID, 
        CASE
            WHEN Raw_Skill LIKE '%collaboration and communication%' THEN 'Team Work'
            WHEN Raw_Skill LIKE '%time management and multitasking%' THEN 'Time Management'
			WHEN Raw_Skill LIKE '%Creativity and design sense%' THEN 'Creativity'
            ELSE Raw_Skill 
        END AS Final_Skill_1,
        CASE
            WHEN Raw_Skill LIKE '%collaboration and communication%' THEN 'Communication'
            WHEN Raw_Skill LIKE '%time management and multitasking%' THEN 'Multitasking'
			WHEN Raw_Skill LIKE '%Creativity and design sense%' THEN 'Design Sense'
            ELSE NULL 
        END AS Final_Skill_2
    FROM 
        Skills_CTE
)
-- Combining results and counting resumes with each skill
SELECT 
    Final_Skill AS Skill, 
    COUNT(DISTINCT ResumeID) AS Skill_Count
FROM (
    SELECT 
        ResumeID, 
        CASE 
            WHEN Final_Skill_1 LIKE 'attention to detail%' THEN 'Attention to details'
            WHEN Final_Skill_1 LIKE 'problem solving%' THEN 'Problem Solving Skills'
            WHEN Final_Skill_1 = 'SQL Server' THEN 'SQL'
            ELSE Final_Skill_1 
        END AS Final_Skill
    FROM Split_Skills_CTE
    WHERE Final_Skill_1 IS NOT NULL

    UNION ALL

    SELECT 
        ResumeID, 
        CASE 
            WHEN Final_Skill_2 LIKE 'attention to detail%' THEN 'Attention to details'
            WHEN Final_Skill_2 LIKE 'problem solving%' THEN 'Problem Solving Skills'
            WHEN Final_Skill_2 = 'SQL Server' THEN 'SQL'
            ELSE Final_Skill_2 
        END AS Final_Skill
    FROM Split_Skills_CTE
    WHERE Final_Skill_2 IS NOT NULL
) AS Combined_Skills
WHERE LTRIM(RTRIM(Final_Skill)) <> '' 
GROUP BY Final_Skill;

SELECT * FROM Skills_list order by Skill_Count desc

--------------------------------------------------------------------------------------------------------------------------------------------

-- View to count the number of skills for each ResumeID

CREATE VIEW skills_count AS
WITH Cleaned_Skills_CTE AS (
    SELECT    -- Cleaning of extracted skills for analysis
        ResumeID, 
        LTRIM(RTRIM(REPLACE(Skill.value, 'non', ''))) AS Cleaned_Skill 
    FROM 
        Resume_data
    CROSS APPLY STRING_SPLIT(Skills, ',') AS Skill 
),
Processed_Skills_CTE AS (
    SELECT 
        ResumeID, 
        CASE
            WHEN Cleaned_Skill LIKE 'attention to detail%' THEN 'Attention to details'
            WHEN Cleaned_Skill LIKE 'problem solving%' THEN 'Problem Solving Skills'
            WHEN Cleaned_Skill = 'SQL Server' THEN 'SQL'
            ELSE Cleaned_Skill 
        END AS Final_Skill
    FROM 
        Cleaned_Skills_CTE
    WHERE LTRIM(RTRIM(Cleaned_Skill)) <> '' 
)
-- Count the number of skills for each ResumeID
SELECT 
    ResumeID, 
    COUNT(DISTINCT Final_Skill) AS Skill_Count
FROM 
    Processed_Skills_CTE
GROUP BY 
    ResumeID;

select * from skills_count

----------------------------------------------------------------------------------------------------------------------------------------------
-- view to abbrevate the education degrees

CREATE VIEW Abbreviated_Education AS
SELECT 
    ResumeID, 
    CASE 
        WHEN Education LIKE '%Bachelor of Engineering%' THEN 'Engineering'
		WHEN Education LIKE '%Bachelor of  Engineering%' THEN 'Engineering'
		WHEN Education LIKE '%Bachelor of Engineering and Technology%' THEN 'Engineering'
        WHEN Education LIKE '%Bachelor of Science%' THEN 'B.Sc'
        WHEN Education LIKE '%Master of Science%' THEN 'M.Sc'
		WHEN Education LIKE '%Master of  Science%' THEN 'M.Sc'
        WHEN Education LIKE '%Master of Business Administration%' THEN 'MBA'
        WHEN Education LIKE '%Doctor of Philosophy%' THEN 'Ph.D'
        WHEN Education LIKE '%Bachelor of Commerce%' THEN 'B.Com'
        ELSE Education  
    END AS Abbreviated_Education
FROM 
    Resume_data;

SELECT * FROM  Abbreviated_Education
