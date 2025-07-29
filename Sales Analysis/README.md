## Description
This Jupyter Notebook conducts a comprehensive sales analysis for a chain of retail stores by loading and merging daily transaction data with store metadata, performing exploratory data analysis to uncover trends and relationships (e.g., the effects of promotions, holidays, and customer counts on revenue), engineering time‑based and promotional features, and finally applying Facebook Prophet to generate and visualize 90‑day sales forecasts for each store.

## Notebook Outline
1. **Imports & Data Loading**  
   - Basic checks for missing values and data types

2. **Data Inspection & Summary Statistics**  
   - Compute overall and per‑store summary statistics (mean, min/max, percentiles)  
   - Visualize missing‑data patterns

3. **Exploratory Data Analysis (EDA)**  
   - Distribution plots for sales and customer counts  
   - Monthly and seasonal sales trends  
   - Impact of Promo1 and Promo2 on sales  
   - Effect of school and state holidays  
   - Comparison across store types and assortments

4. **Data Merging & Feature Engineering**  
   - Merge sales and store datasets  
   - Extract date features (year, month, day of week)  
   - Encode promotional intervals (Promo2 start/end)  
   - Compute correlation matrix and visualize via heatmap

5. **Time Series Forecasting with Facebook Prophet**  
   - Prepare holiday DataFrame for Prophet  
   - Fit a Prophet model per store  
   - Generate 90‑day future DataFrame  
   - Plot historical vs. forecasted sales and components (trend, weekly/yearly seasonality)
