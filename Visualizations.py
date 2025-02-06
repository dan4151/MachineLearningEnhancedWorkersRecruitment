# Convert numerical features to Pandas
features = ["moreThanYear", "avg_past_months", "num_education", "num_past_experience"]
pdf = df_filled.select(features).toPandas()

# Create overlapping histograms
plt.figure(figsize=(12, 8))
for i, feature in enumerate(features[1:], 1):  # Skip "moreThanYear"
    plt.subplot(2, 2, i)
    plt.hist(pdf[pdf["moreThanYear"] == 0][feature], bins=30, alpha=0.5, label="Left (<1 Year)", color="red", edgecolor="black")
    plt.hist(pdf[pdf["moreThanYear"] == 1][feature], bins=30, alpha=0.5, label="Stayed (≥1 Year)", color="blue", edgecolor="black")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {feature} by Retention")
    plt.legend()
plt.tight_layout()
plt.show()

# Convert job title data to Pandas
pdf = df_filled.select("title", "moreThanYear").toPandas()

# Get top 10 most common job titles
top_titles = pdf["title"].value_counts().nlargest(10).index
pdf_filtered = pdf[pdf["title"].isin(top_titles)]

# Create stacked bar chart
plt.figure(figsize=(10, 5))
sns.countplot(data=pdf_filtered, x="title", hue="moreThanYear", palette=["#FF9999", "#66B3FF"])
plt.xlabel("Job Title")
plt.ylabel("Count")
plt.title("Job Title vs. Retention")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Stayed More Than a Year", labels=["No (0)", "Yes (1)"])
plt.show()

# Convert Spark DataFrame to Pandas
pdf = df_filled.select([
    "moreThanYear", "mean_acceptance_rate", "mean_graduation_rate",
    "mean_avg_cost_after_aid", "avg_past_months", "recommendations_count",
    "num_education", "num_past_experience"
]).toPandas()

# Compute correlation matrix
corr_matrix = pdf.corr()

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Import necessary libraries
import numpy as np

# Given feature names and coefficients (Replace with actual model coefficients)
feature_names = [
    "num_education", "avg_past_months", "num_past_experience",
    "mean_graduation_rate", "recommendations_count",
    "mean_acceptance_rate", "mean_avg_cost_after_aid"  # Swapped order
]

# Adjusted coefficients (Replace with real values from your model)
coefficients = [0.4, 0.35, 0.3, 0.25, 0.2, -0.1, -0.15]

# Create a DataFrame for visualization
feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

# Sort by absolute importance
feature_importance_df["Absolute Coefficient"] = feature_importance_df["Coefficient"].abs()
feature_importance_df = feature_importance_df.sort_values(by="Absolute Coefficient", ascending=False)

# Plot feature importance
plt.figure(figsize=(8, 5))
plt.barh(feature_importance_df["Feature"], feature_importance_df["Coefficient"],
         color=np.where(feature_importance_df["Coefficient"] > 0, "steelblue", "salmon"))
plt.axvline(0, color="black", linestyle="--")  # Reference line at 0
plt.xlabel("Feature Importance (Coefficient Value)")
plt.ylabel("Feature")
plt.title("Feature Importance in Logistic Regression")
plt.gca().invert_yaxis()  # Invert y-axis for better readability
plt.show()
