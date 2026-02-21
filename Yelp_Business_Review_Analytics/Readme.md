# Yelp Business Review Analytics (AWS S3 + Snowflake + Python + SQL)
![Concept Diagram](Concept.JPG)

End to end semi structured analytics pipeline on the Yelp Academic Dataset. I split a ~5GB reviews JSONL file into upload friendly shards using Python, staged raw JSON in AWS S3, bulk loaded into Snowflake as VARIANT, flattened into relational tables, enriched each review with sentiment via a Snowflake Python UDF (TextBlob polarity), and answered business questions using SQL analytics.

## Architecture
See `Concept.JPG` for the pipeline diagram:
Yelp Reviews JSONL → Python file splitting → S3 → Snowflake raw VARIANT tables → flattening into tables → Python UDF sentiment → SQL analytics

## Dataset
- Reviews: `yelp_academic_dataset_review.json` (JSON lines, ~5GB, ~7M records)
- Businesses: `yelp_academic_dataset_business.json` (~100MB)

## Repo contents
- `File_split.ipynb`  
  Splits the large Yelp reviews JSONL into `split_file_1.json` … `split_file_25.json`
- `yelp_rev_tbl.sql`  
  Creates raw + flattened review tables. Imports data from AWS S3 and applies the sentiment UDF
- `yelp_business_tbl.sql`  
  Creates raw + flattened business tables. Imports data from AWS S3.
- `UDF.sql`  
  Creates the `analyze_sentiment` Snowflake Python UDF (TextBlob)
- `Analysis.sql`  
  Example analytics queries (categories, reviewers, recency, seasonality, 5-star share, city rankings, rating stability, positive sentiment leaders)
- `Concept.JPG`  
  Pipeline map

## Prerequisites
- AWS account and an S3 bucket 
- Snowflake account with a database, schema, and warehouse you can use
- Python (Jupyter) to run `File_split.ipynb`
