-- Analysis starts
-- number of businesses from each category
with cte as (
select business_id, trim(A.value) as category
from tbl_yelp_businesses
,lateral split_to_table(categories,',') A
)
select category,count(*) as no_of_business
from cte
group by 1
order by 2 desc

-- Restaurant comes as the highest number. So find top 10 users who have reviewed the most businesses in the Restaurant Category

select r.user_id, count(distinct r.business_id)
from tbl_yelp_reviews r
inner join tbl_yelp_businesses b on r.business_id=b.business_id
where b.categories ilike '%restaurant%'
group by 1
order by 2 desc
limit 10

-- Analysing most popular categories of businesses (based on the number of reviews)
with cte as (
select business_id, trim(A.value) as category
from tbl_yelp_businesses
,lateral split_to_table(categories,',') A
)
select category, count(*) as no_of_reviews
from cte
inner join tbl_yelp_reviews r on cte.business_id = r.business_id
group by 1
order by 2 desc

-- Analysing 3 most recent reviews for each businesses
with cte as (
select r.*, b.name
,row_number() over(partition by r.business_id order by review_date desc) as rn
from tbl_yelp_reviews r
inner join tbl_yelp_businesses b on r.business_id=b.business_id
)
select * from cte
where rn<=3

-- Analysing the months with the highest number of reviews
select month(review_date) as review_month, count(*) as no_of_reviews
from tbl_yelp_reviews
group by 1
order by 2 desc

-- Fiding percentage of 5* reviews for each business
select b.business_id, b.name, count(*) as total_reviews
,sum(case when r.review_stars=5 then 1 else 0 end) as star5_reviews
,star5_reviews*100/total_reviews as percent_5_star
from tbl_yelp_reviews r
inner join tbl_yelp_businesses b on r.business_id = b.business_id
group by 1,2
order by 4 desc

-- Finding top 5 most reviewed businesses in each city
with cte as (
select b.city, b.business_id, b.name, count(*) as total_reviews
from tbl_yelp_reviews r
inner join tbl_yelp_businesses b on r.business_id = b.business_id
group by 1,2,3
)
select *
from cte
qualify row_number() over (partition by city order by total_reviews desc) <=5

-- Finding average rating of businesses that have atleast 100 reviews
select b.business_id, b.name, count(*) as total_reviews
,avg(review_stars) as avg_rating
from tbl_yelp_reviews r
inner join tbl_yelp_businesses b on r.business_id = b.business_id
group by 1,2
having count(*)>=100

-- Listing the top 10 users who have written the most reviews, along with the businesses they reviewed
with cte as (
select r.user_id, count(*) as total_reviews
from tbl_yelp_reviews r
inner join tbl_yelp_businesses b on r.business_id = b.business_id
group by 1
order by 2 desc
limit 10
)
select user_id, business_id
from tbl_yelp_reviews where user_id in (select user_id from cte)
group by 1,2
order by user_id

-- Finding top 10 businesses with highest positive sentiment reviews
select r.business_id,b.name, count(*) as total_reviews
from tbl_yelp_reviews r
inner join tbl_yelp_businesses b on r.business_id = b.business_id
where sentiments = 'Positive'
group by 1,2
order by 3 desc
limit 10