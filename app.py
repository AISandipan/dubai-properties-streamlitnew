# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO

# Load and cache data
@st.cache_data
def load_data():
    df = pd.read_csv("Dubaiproperties_data.csv")
    df.columns = df.columns.str.strip()
    df.fillna("", inplace=True)
    return df

df = load_data()

# Sidebar navigation
st.sidebar.title("🏙️ Dubai Property App")
view = st.sidebar.radio("Go to", [
    "📄 View & Filter Data",
    "📊 EDA Profiling",
    "🏘️ Neighborhood Summary",
    "🏡 Property Recommendations",
    "📥 Download Report"
])

# View & Filter
if view == "📄 View & Filter Data":
    st.title("🔎 View & Filter Properties")
    neighborhood = st.selectbox("Select Neighborhood", ["All"] + sorted(df["neighborhood"].unique()))
    quality = st.selectbox("Select Quality", ["All"] + sorted(df["quality"].unique()))

    filtered = df.copy()
    if neighborhood != "All":
        filtered = filtered[filtered["neighborhood"] == neighborhood]
    if quality != "All":
        filtered = filtered[filtered["quality"] == quality]

    st.dataframe(filtered)

# EDA Report
elif view == "📊 EDA Profiling":
    st.title("📊 Exploratory Data Analysis Report")

    profile = ProfileReport(df, title="Dubai Properties EDA Report", explorative=True)
    st_profile_report(profile)

    st.subheader("📈 Price Distribution")
    fig1 = px.histogram(df, x="price", nbins=50, title="Price Distribution")
    st.plotly_chart(fig1)

    st.subheader("🏡 Bedroom Count Distribution")
    fig2 = px.bar(df["no_of_bedrooms"].value_counts().reset_index(), 
                  x="index", y="no_of_bedrooms", 
                  labels={"index": "No. of Bedrooms", "no_of_bedrooms": "Count"},
                  title="Distribution of Bedrooms")
    st.plotly_chart(fig2)

# Summary Statistics
elif view == "🏘️ Neighborhood Summary":
    st.title("📍 Neighborhood Insights")

    st.subheader("💰 Top 10 Neighborhoods by Average Price")
    top_neighborhoods = df.groupby("neighborhood")["price"].mean().sort_values(ascending=False).head(10).reset_index()
    st.dataframe(top_neighborhoods)

    fig_top = px.bar(top_neighborhoods, x="neighborhood", y="price", title="Top 10 Neighborhoods by Avg Price")
    st.plotly_chart(fig_top)

    st.subheader("🛏️ Bedroom Distribution")
    bedrooms = df["no_of_bedrooms"].value_counts().reset_index()
    bedrooms.columns = ["No. of Bedrooms", "Count"]
    st.dataframe(bedrooms)

    fig_bedrooms = px.pie(bedrooms, names="No. of Bedrooms", values="Count", title="Bedroom Distribution")
    st.plotly_chart(fig_bedrooms)

    st.subheader("🏷️ Property Quality Distribution")
    quality = df["quality"].value_counts().reset_index()
    quality.columns = ["Quality", "Count"]
    st.dataframe(quality)

    fig_quality = px.bar(quality, x="Quality", y="Count", title="Property Quality Distribution")
    st.plotly_chart(fig_quality)

# Recommendations
elif view == "🏡 Property Recommendations":
    st.title("🏡 Similar Property Recommendations")

    features = ["latitude", "longitude", "price", "size_in_sqft", "price_per_sqft", "no_of_bedrooms", "no_of_bathrooms"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[features])
    sim_matrix = cosine_similarity(scaled)

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

    selected_id = st.selectbox("Choose Property ID", df["id"])
    idx = df[df["id"] == selected_id].index[0]
    similar_idxs = sorted(list(enumerate(sim_matrix[idx])), key=lambda x: x[1], reverse=True)[1:6]  # Top 5
    recommendations = explain_similarity(idx, df, [i[0] for i in similar_idxs])

    st.write("🔁 Top 5 Similar Properties with Reasons:")
    for rec in recommendations:
        st.markdown(f"- {rec}")

# Download Report
elif view == "📥 Download Report":
    st.title("📦 Download Excel Report")

    # Recompute recommendations with explanation
    features = ["latitude", "longitude", "price", "size_in_sqft", "price_per_sqft", "no_of_bedrooms", "no_of_bathrooms"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[features])
    sim_matrix = cosine_similarity(scaled)

    def explain_all(i):
        base = df.iloc[i]
        similar_idxs = [j[0] for j in sorted(list(enumerate(sim_matrix[i])), key=lambda x: x[1], reverse=True)[1:4]]
        return "; ".join(explain_similarity(i, df, similar_idxs))

    df["Recommended Properties"] = df.index.map(lambda i: explain_all(i))

    # Generate summaries
    top_neighborhoods = df.groupby("neighborhood")["price"].mean().sort_values(ascending=False).head(10).reset_index()
    bedroom_summary = df["no_of_bedrooms"].value_counts().reset_index()
    bedroom_summary.columns = ["No. of Bedrooms", "Property Count"]
    quality_summary = df["quality"].value_counts().reset_index()
    quality_summary.columns = ["Quality", "Count"]

    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Properties + Recommendations", index=False)
        top_neighborhoods.to_excel(writer, sheet_name="Top Neighborhoods", index=False)
        bedroom_summary.to_excel(writer, sheet_name="Bedroom Summary", index=False)
        quality_summary.to_excel(writer, sheet_name="Quality Summary", index=False)
    output.seek(0)

    st.download_button(
        label="📥 Download Excel Report",
        data=output,
        file_name="dubai_property_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
