-- Q1 WHERE then HAVING
SELECT customer_id,SUM(order_amount) total FROM orders WHERE status='completed' AND order_amount>500 GROUP BY customer_id HAVING SUM(order_amount)>8000;
-- Q2 three-table join
SELECT c.city,p.category,SUM(o.order_amount) revenue FROM orders o JOIN customers c USING(customer_id) JOIN products p USING(product_id) WHERE o.status='completed' GROUP BY c.city,p.category ORDER BY c.city,p.category;
-- Q3 scalar subquery
SELECT product_id,product_name,price_inr FROM products WHERE price_inr>(SELECT AVG(price_inr) FROM products);
-- Q4 LEFT JOIN
SELECT c.customer_id FROM customers c LEFT JOIN orders o ON c.customer_id=o.customer_id AND o.status='completed' WHERE o.order_id IS NULL;
-- Q5 two CTEs
WITH spend AS (SELECT customer_id,SUM(order_amount) total FROM orders WHERE status='completed' GROUP BY customer_id), avg_spend AS (SELECT AVG(total) avg_total FROM spend) SELECT spend.customer_id,total FROM spend,avg_spend WHERE total>avg_total;
-- Q6 ranking windows
WITH spend AS (SELECT c.city,o.customer_id,SUM(o.order_amount) total FROM orders o JOIN customers c USING(customer_id) WHERE o.status='completed' GROUP BY c.city,o.customer_id) SELECT city,customer_id,total,RANK() OVER(PARTITION BY city ORDER BY total DESC) rnk,DENSE_RANK() OVER(PARTITION BY city ORDER BY total DESC) dense_rnk,ROW_NUMBER() OVER(PARTITION BY city ORDER BY total DESC,customer_id) row_num FROM spend;
-- Q7 running total
WITH daily AS (SELECT substr(order_ts,1,10) day,SUM(order_amount) revenue FROM orders WHERE status='completed' GROUP BY substr(order_ts,1,10)) SELECT day,revenue,SUM(revenue) OVER(ORDER BY day) cumulative_revenue FROM daily;