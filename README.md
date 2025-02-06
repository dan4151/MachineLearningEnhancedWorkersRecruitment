# 👋 About Us
Hi, We are Karni Shifrin, Idan Perez, and Dan Amler, Data Engineering and Information students from the Technion with a passion for AI-driven decision-making and big data analytics. This project, Machine Learning Enhanced Workers Recruitment, was developed as part of our Data Collection Lab course, where we explored the power of PySpark, Databricks, and Machine Learning to optimize hiring decisions. Our model predicts employee retention, helping companies make smarter recruitment choices based on structured insights from LinkedIn and university data. 🚀  

## 🚀 Project: Machine Learning Enhanced Workers Recruitment  
### 🔍 Overview  
Hiring the right employees is one of the biggest challenges for companies. Bad hires lead to wasted time and money, while good hires drive growth and innovation. My project aims to predict whether a candidate will stay in a company for more than a year, using LinkedIn and university data.  

### 📊 Data Collection & Processing  
#### 📄 Data Sources:  
LinkedIn company and employee data from Bright Data, web-scraped university rankings (Google & Selenium).  
#### 🔍 Data Processing:  
Extracted job experience details using PySpark.  
Mapped companies based on size to filter out unreliable data.  
Integrated education details (Acceptance Rate, Graduation Rate, Cost After Aid) into the dataset.  
### 📈 Machine Learning Approach  
Feature Engineering:  
Extracted job history (duration, number of past jobs).  
Mapped education details to employment trends.  
Embedded textual features using pre-trained word embeddings.  
Modeling:  
Used Logistic Regression with class weighting to handle data imbalance (72% of employees stayed >1 year).  
Evaluated using AUC (71.8%) and F1-Score (74.72%).  
Key insights:  
More LinkedIn recommendations → Higher retention.  
Higher graduation rates → Stronger job stability.  
Higher tuition costs → Lower retention (financial independence?).  
### 🛠 How to Use the Code  
This repository contains our Machine Learning Enhanced Workers Recruitment project, implemented using PySpark and Databricks. The code is structured into reusable classes for data processing, feature engineering, and machine learning modeling. The Databricks note book is also added to this repository, you can run the cells by their order. 
#### 1️⃣ Setup  
Before running the code, ensure you have:  
A Databricks environment or local PySpark setup.  
Access to the LinkedIn datasets (linkedin_people_train_data and linkedin_train_data).  
A CSV file containing university rankings.  

#### 2️⃣ Running the Data Processing Pipeline  
The LinkedInDataProcessor class handles:  
Data loading (.parquet and .csv).  
Feature extraction (experience duration, job roles, education details).  
Data cleaning and transformation.  
The SparkNLPPreprocessorStatic uses pre-trained embeddings on the texual features.
You can choose which features are used in the final structured data. 
``` python
spark = SparkSession.builder.appName("LinkedInDataProcessing").getOrCreate()
processor = LinkedInDataProcessor(spark, "/path/to/companies", "/path/to/employees", "/path/to/universities.csv")
processed_data = processor.run_pipeline()
preprocessor = SparkNLPPreprocessorStatic(
        text_cols=["title", "subtitle"],
        numeric_cols=["mean_acceptance_rate", "mean_graduation_rate", "mean_avg_cost_after_aid", "num_past_experience",
                      "num_education", "recommendations_count", "avg_past_months"]
    )

    preprocessor.build_pipeline()
    final_df = preprocessor.fit_transform(processed_data)
    final_df = final_df.filter(
        "title IS NOT NULL AND TRIM(title) != '' "
        "AND subtitle IS NOT NULL AND TRIM(subtitle) != '' "
    )
```
#### 3️⃣ Training the Machine Learning Model
```python
model = LinkedInLogisticModel(spark, final_df)
metrics = model.run_pipeline()
```

### 💡 Skills & Tech Stack  
✅ Big Data Processing: PySpark, Databricks  
✅ Machine Learning: MLlib, Logistic Regression  
✅ Data Scraping & Integration: Selenium, Web Scraping  
✅ Feature Engineering: Text embeddings, numerical feature extraction  
✅ Visualization & Analytics: Matplotlib, Data Analysis  

Let us know if you'd like any refinements or additional details! 🚀  
