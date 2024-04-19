# URL_Citation_Classification_Intermediate
The source code for the paper "On an Intermediate Task for Classifying URL Citations on Scholarly Papers" in LREC-COLING 2024

# Source Code
All programs used in experiments are in ```/src``` directory.
- training.py: Main component for training the model (used for both simple fine-tuning and our method)
- *_run.py: The program to load and preprocess the dataset of each task (e.g., cola_run.py -> The program for CoLA)
  - url_cite_run.py: Main script for our method and Tsunokake and Matsubara (2022)'s method
  - url_zhao_run.py: The script for Zhao et al. (2019)'s method
  - others: Scripts for Section "4.3.5. Effectiveness of Our Method for Other Text Classification Tasks"
