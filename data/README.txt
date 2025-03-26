test_data.csv is the complete dataset with the following columns:

student_id = A unique student identifier for each student who took the test

item_id = A unique question identifier for each question in the dataset. Note that there are 117 questions in total.

discrimination = Discrimination parameter for each question

difficulty = Difficulty parameter for each item

response = A binary response value (1: correct; 0 = incorrect)

response_time = Time spent on each question (in seconds; max 6 seconds per question)

final_score = Students' final scores based on the entire set of questions (i.e., 117 questions in total)

Note (1): This is a long-format dataset where there are multiple rows for each student. In each row, you will view which item was answered, response (1 or 0), response_time, discrimination and difficulty parameters, and the student's final score. Since the final score was calculated based on the entire set of questions, all "final_score" values for a given student will be the same. 
We will use final_score as the ground truth for testing the accuracy of our RL-based adaptive testing system.

Note (2): We will use response, difficulty, and discrimination to estimate the ability for a vector of responses. We will use difficulty, discrimination, a student's interim ability estimate during the test to calculate Fisher information for all questions in the pool and try to select questions with high Fisher information (but also take the response time into account). 

Note (3): sample_data.csv has data for a single student randomly selected from test_data.csv. I used this small dataset to demonstrate the ability estimation and Fisher information functions in this folder. 

Note (4): Scoring_and_Fisher_Information_Functions.ipynb contains the functions that you will need to perform scoring (i.e., ability estimation) and compute Fisher information for the questions. 

Our goal is to build an adaptive test with 20 questions that can be completed within 60 seconds. We can play with these values to make the case more or less extreme. 