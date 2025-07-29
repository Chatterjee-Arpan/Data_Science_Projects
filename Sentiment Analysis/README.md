## Notebook Outline
1. **Imports & Data Loading**  
   - Import libraries (pandas, NumPy, Seaborn, Matplotlib, NLTK, scikit‑learn, WordCloud)  
   - Read `amazon_alexa.tsv` into a DataFrame

2. **Field Information**  
   - Define and explain each column:  
     - `rating` (1–5), `date`, `variation` (Echo model), `verified_reviews` (text), `feedback` (binary sentiment)

3. **Initial Preprocessing & Feature Engineering**  
   - Fill missing review texts with empty strings  
   - Create a `length` column (character count of each review)  
   - Display summary statistics and plot a histogram of review lengths  
   - Show the longest review instance

4. **Exploratory Data Analysis (EDA)**  
   - Plot distributions of `rating` and `feedback` (positive vs. negative)  
   - Barplot of average rating by `variation`  
   - Generate a WordCloud of all reviews  
   - Generate a WordCloud of negative reviews

5. **Text Cleaning Pipeline**  
   - Download NLTK stopwords  
   - Define `message_clean()` to remove punctuation and English stopwords

6. **Text Vectorization**  
   - Initialize `CountVectorizer` with the cleaning function as the analyzer  
   - Fit and transform the entire review corpus  
   - Inspect feature names and vectorized array shape

7. **Dataset Preparation for Modeling**  
   - Drop the original `verified_reviews` column  
   - Build a DataFrame from the vectorized features  
   - Concatenate feature DataFrame with original metadata  
   - Separate into feature matrix `X` and target vector `y`

8. **Train‑Test Split**  
   - Split `X` and `y` into training and testing sets (80/20)

9. **Model Training & Evaluation: Multinomial Naive Bayes**  
   - Train `MultinomialNB` on the training set  
   - Predict on train and test sets  
   - Display classification reports and confusion matrices (heatmaps)

10. **Model Training & Evaluation: Logistic Regression**  
    - Train `LogisticRegression`  
    - Predict on the test set  
    - Compute and display accuracy score and confusion matrix heatmap
