# -*- coding: utf-8 -*-
"""

Report 
"""

#!pip install openpyxl

#!pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip

import pandas as pd
from pandas_profiling import ProfileReport
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
file_path = "Rightbite_sample_data.xlsx"  # Update with your file path
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Generate EDA Report
profile = ProfileReport(df, title="Kitchen & Product Improvement Report", explorative=True)
profile.to_file("kitchen_product_report.html")

# Data Preprocessing
df.fillna("", inplace=True)  # Fill NaN values with empty strings

# Feature Engineering: Combining relevant text columns for analysis
df["combined_text"] = df["Complaint Details"] + " " + df["Affected Items"]

# Vectorization using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df["combined_text"])

# Compute Similarity Scores
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(index, df, similarity_matrix, top_n=3):
    """Recommends top kitchens or products based on similar complaints."""
    scores = list(enumerate(similarity_matrix[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    recommendations = [df.iloc[i[0]]["Kitchen Name"] for i in scores]
    return recommendations

# Identify kitchens with low ratings and common complaints
low_rated_kitchens = df[df["Rating"] <= 2]
complaint_analysis = low_rated_kitchens.groupby("Kitchen Name")["Complaint Details"].apply(lambda x: ", ".join(x)).reset_index()
complaint_analysis["Suggested Improvements"] = complaint_analysis["Complaint Details"].apply(
    lambda x: "Improve portion size" if "portion" in x.lower() else
              "Ensure food is fresh and hot" if "cold" in x.lower() else
              "Follow cooking instructions properly" if "instruction" in x.lower() else
              "General quality check required"
)

# Save improvements to CSV
complaint_analysis.to_csv("kitchen_improvements.csv", index=False)

# Example Usage: Get recommendations for first entry
example_index = 0
recommendations = get_recommendations(example_index, df, cosine_sim)
print(f"Recommended Kitchens for Improvement based on complaints: {recommendations}")

# Save recommendations to a CSV file
df["Recommended Kitchens"] = df.index.map(lambda i: get_recommendations(i, df, cosine_sim))
df.to_csv("recommended_kitchens.csv", index=False)

print("EDA report saved as 'kitchen_product_report.html', recommendations saved as 'recommended_kitchens.csv', and improvements saved as 'kitchen_improvements.csv'")

#!pip install openpyxl

import pandas as pd

# Step 1: Load the Excel file
file_path = "Aug - Rightbite_Sample Data.xlsx"  # Ensure this file is in the same directory as the script
xls = pd.ExcelFile(file_path)

# Step 2: Load the sheet into a DataFrame
df = pd.read_excel(xls, sheet_name="Aug 2025")

# Step 3: Data Cleaning - Trim spaces and standardize column names
df.columns = df.columns.str.strip()

# Step 4: Split multiple items in "Affected Items" into separate rows
df = df.assign(Affected_Items=df["Affected Items"].str.split(",")) \
       .explode("Affected_Items")

# Step 5: Trim spaces from food item names
df["Affected_Items"] = df["Affected_Items"].str.strip()

# Step 6: Compute frequency of complaints per food item
food_complaint_counts = df["Affected_Items"].value_counts()

# Step 7: Compute most frequent complaint category per food item
complaint_category_counts = df.groupby("Affected_Items")["Incident/ Review Category"].agg(lambda x: x.value_counts().idxmax())

# Step 8: Combine both metrics into a summary DataFrame
food_complaint_summary = pd.DataFrame({
    "Food Item": food_complaint_counts.index,
    "Complaint Count": food_complaint_counts.values,
    "Most Frequent Complaint Category": complaint_category_counts.values
})

# Step 9: Analyze complaints based on kitchen location for the targeted food items
kitchen_analysis = df.groupby(["Kitchen Name", "Affected_Items"])["Incident/ Review Category"].count().reset_index()
kitchen_analysis = kitchen_analysis.rename(columns={"Incident/ Review Category": "Complaint Count"})

# Step 10: Define output file path
output_file_path = "food_complaint_analysis.xlsx"

# Step 11: Save both summaries to an Excel file
with pd.ExcelWriter(output_file_path) as writer:
    food_complaint_summary.to_excel(writer, sheet_name="Food Complaint Summary", index=False)
    kitchen_analysis.to_excel(writer, sheet_name="Kitchen Complaint Analysis", index=False)

print(f"Report saved as {output_file_path}")

#!pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip

import pandas as pd

# Step 1: Load the Excel file
file_path = "Aug - Rightbite_Sample Data.xlsx"  # Ensure this file is in the same directory as the script
xls = pd.ExcelFile(file_path)

# Step 2: Load the sheet into a DataFrame
df = pd.read_excel(xls, sheet_name="Aug 2025")

# Step 3: Data Cleaning - Trim spaces and standardize column names
df.columns = df.columns.str.strip()

# Step 4: Split multiple items in "Affected Items" into separate rows
df = df.assign(Affected_Items=df["Affected Items"].str.split(",")) \
       .explode("Affected_Items")

# Step 5: Trim spaces from food item names
df["Affected_Items"] = df["Affected_Items"].str.strip()

# Step 6: Compute frequency of complaints per food item
food_complaint_counts = df["Affected_Items"].value_counts()

# Step 7: Compute most frequent complaint category per food item
complaint_category_counts = df.groupby("Affected_Items")["Incident/ Review Category"].agg(lambda x: x.value_counts().idxmax())

# Step 8: Aggregate complaint names per food item
complaint_names = df.groupby("Affected_Items")["Incident/ Review Category"].unique().apply(lambda x: ", ".join(x))

# Step 9: Combine metrics into a summary DataFrame
food_complaint_summary = pd.DataFrame({
    "Food Item": food_complaint_counts.index,
    "Complaint Count": food_complaint_counts.values,
    "Most Frequent Complaint Category": complaint_category_counts.values,
    "Complaint Names": complaint_names.values
})

# Step 10: Sort the summary to get top 20 food items with most complaints
food_complaint_summary = food_complaint_summary.head(20)

# Step 11: Analyze complaints based on kitchen location for the targeted food items
kitchen_analysis = df.groupby(["Kitchen Name", "Affected_Items", "Incident/ Review Category"]).size().reset_index(name="Complaint Count")

# Step 12: Define output file path
output_file_path = "food_complaint_analysis.xlsx"

# Step 13: Save both summaries to an Excel file
with pd.ExcelWriter(output_file_path) as writer:
    food_complaint_summary.to_excel(writer, sheet_name="Food Complaint Summary", index=False)
    kitchen_analysis.to_excel(writer, sheet_name="Kitchen Complaint Analysis", index=False)

print(f"Report saved as {output_file_path}")

import pandas as pd

# Step 1: Load the Excel file
file_path = "Aug - Rightbite_Sample Data.xlsx"
# Ensure this file is in the same directory as the script
xls = pd.ExcelFile(file_path)

# Step 2: Load the sheet into a DataFrame
df = pd.read_excel(xls, sheet_name="Aug 2025")

# Step 3: Data Cleaning - Trim spaces and standardize column names
df.columns = df.columns.str.strip()

# Step 4: Split multiple items in "Affected Items" into separate rows
df = df.assign(Affected_Items=df["Affected Items"].str.split(",")) \
       .explode("Affected_Items")

# Step 5: Trim spaces from food item names
df["Affected_Items"] = df["Affected_Items"].str.strip()

# Step 6: Add "Complaint Name" column (same as "Incident/ Review Category")
df["Complaint Name"] = df["Incident/ Review Category"]

# Step 7: Compute frequency of complaints per food item
food_complaint_counts = df["Affected_Items"].value_counts()

# Step 8: Compute most frequent complaint category per food item
complaint_category_counts = df.groupby("Affected_Items")["Complaint Name"].agg(lambda x: x.value_counts().idxmax())

# Step 9: Combine both metrics into a summary DataFrame
food_complaint_summary = pd.DataFrame({
    "Food Item": food_complaint_counts.index,
    "Complaint Count": food_complaint_counts.values,
    "Most Frequent Complaint Category": complaint_category_counts.values
})

# Step 10: Analyze complaints based on kitchen location for the targeted food items
kitchen_analysis = df.groupby(["Kitchen Name", "Affected_Items", "Complaint Name"]).size().reset_index(name="Complaint Count")

# Step 11: Define output file path
output_file_path = "food_complaint_analysis.xlsx"

# Step 12: Save both summaries to an Excel file
with pd.ExcelWriter(output_file_path) as writer:
    food_complaint_summary.to_excel(writer, sheet_name="Food Complaint Summary", index=False)
    kitchen_analysis.to_excel(writer, sheet_name="Kitchen Complaint Analysis", index=False)

print(f"Report saved as {output_file_path}")

