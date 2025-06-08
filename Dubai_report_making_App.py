# -*- coding: utf-8 -*-
"""
Dubai Properties - Analysis & Recommendation Report
"""

import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# STEP 1: Load and Inspect Data
# -------------------------------
df = pd.read_csv("Dubaiproperties_data.csv")
df.columns = df.columns.str.strip()
df.fillna("", inplace=True)

# -------------------------------
# STEP 2: Generate EDA Report
# -------------------------------
eda_profile = ProfileReport(df, title="Dubai Properties EDA Report", explorative=True)
eda_profile.to_file("dubai_properties_eda.html")

# -------------------------------
# STEP 3: Property Similarity
# -------------------------------
features = [
    "latitude", "longitude", "price", "size_in_sqft",
    "price_per_sqft", "no_of_bedrooms", "no_of_bathrooms"
]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])
similarity_matrix = cosine_similarity(scaled_features)

def explain_similarity(base_idx, df, similar_idxs):
    base = df.iloc[base_idx]
    explanations = []
    for idx in similar_idxs:
        sim_prop = df.iloc[idx]
        reason = []
        if abs(base["price"] - sim_prop["price"]) < 50000:
            reason.append("similar price")
        if abs(base["size_in_sqft"] - sim_prop["size_in_sqft"]) < 200:
            reason.append("similar size")
        if abs(base["latitude"] - sim_prop["latitude"]) < 0.01 and abs(base["longitude"] - sim_prop["longitude"]) < 0.01:
            reason.append("nearby location")
        explanations.append(f"{sim_prop['id']} ({', '.join(reason)})")
    return explanations

df["Recommended Properties"] = df.index.map(
    lambda i: "; ".join(explain_similarity(
        i, df,
        [j[0] for j in sorted(list(enumerate(similarity_matrix[i])), key=lambda x: x[1], reverse=True)[1:4]]
    ))
)

# -------------------------------
# STEP 4: Summary Statistics
# -------------------------------
top_neighborhoods = df.groupby("neighborhood")["price"].mean().sort_values(ascending=False).head(10).reset_index()
top_neighborhoods.columns = ["Neighborhood", "Average Price"]

bedroom_summary = df["no_of_bedrooms"].value_counts().reset_index()
bedroom_summary.columns = ["No. of Bedrooms", "Property Count"]

quality_summary = df["quality"].value_counts().reset_index()
quality_summary.columns = ["Quality", "Count"]

# -------------------------------
# STEP 5: Save All Outputs
# -------------------------------
output_file = "dubai_property_report.xlsx"
with pd.ExcelWriter(output_file) as writer:
    df.to_excel(writer, sheet_name="Properties + Recommendations", index=False)
    top_neighborhoods.to_excel(writer, sheet_name="Top Neighborhoods", index=False)
    bedroom_summary.to_excel(writer, sheet_name="Bedroom Summary", index=False)
    quality_summary.to_excel(writer, sheet_name="Quality Summary", index=False)

print(f"✅ Reports saved:\n- EDA: dubai_properties_eda.html\n- Excel: {output_file}")
