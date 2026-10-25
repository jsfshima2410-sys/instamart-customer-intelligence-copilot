# Instamart Customer Intelligence & Support Copilot

## Overview

This repository builds a four-part, reproducible customer intelligence and support copilot over fictional synthetic Instamart-like data. Part A cleans the generated exports and stores trusted tables in CSV and SQLite. Part B segments customers and estimates lapse risk. Part C redacts and classifies support tickets, validates model replies and routes work. Part D retrieves policy passages and runs a tool-using refund agent. All model calls use the supplied offline `mock_llm.py`; no API key or live model is used. All amounts are in INR.

## Reproduce

Run these commands from the repository root in a clean Python 3.11+ environment. The activation command shown is for Windows PowerShell; on macOS/Linux use `source .venv/bin/activate` instead.

1. `python -m venv .venv`
2. `.venv\Scripts\Activate.ps1`
3. `python -m pip install -r requirements.txt`
4. `python -m pip freeze > requirements.txt`
5. `python run_pipeline.py`

`run_pipeline.py` runs the supplied deterministic generator followed by Parts A, B, C and D. Part A prints `ALL CHECKPOINTS PASSED`. The generator's final line is `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. The fixed seeds are in the stage scripts; model calls remain offline.

## Part A

### Raw audit

| Table | Shape | pandas dtypes | Missing by column: count (percent) |
|---|---:|---|---|
| customers | 1,200 × 5 | customer_id object; city object; signup_date object; age string; gender object | customer_id 0 (0%); city 0 (0%); signup_date 0 (0%); age 61 (5.08%); gender 30 (2.50%) |
| products | 40 × 4 | product_id object; product_name object; category object; price_inr int64 | product_id 0 (0%); product_name 0 (0%); category 0 (0%); price_inr 0 (0%) |
| orders | 13,084 × 9 | order_id object; customer_id object; product_id object; quantity int64; order_ts object; status object; discount_pct int64; order_amount object; delivery_minutes float64 | order_id 0 (0%); customer_id 0 (0%); product_id 0 (0%); quantity 0 (0%); order_ts 0 (0%); status 0 (0%); discount_pct 0 (0%); order_amount 53 (0.41%); delivery_minutes 2,541 (19.42%) |
| tickets | 164 × 5 | ticket_id object; customer_id object; order_id object; text object; true_category object | ticket_id 0 (0%); customer_id 0 (0%); order_id 0 (0%); text 0 (0%); true_category 0 (0%). `keep_default_na=False` preserves four blank text strings. |

| Raw audit check | Count |
|---|---:|
| Order rows / exact duplicate rows / pending rows / distinct order IDs | 13,084 / 119 / 656 / 12,309 |
| Unparseable order amounts / negative delivery minutes / blank delivery minutes | 120 / 106 / 2,541 |
| Blank customer ages / ages outside 10–100 / raw city spellings | 61 / 6 / 24 |
| Distinct nonblank raw gender spellings / blank genders | 10 / 30 |

### Cleaning, storage and validation

- Customers: trimmed city strings and applied title case across 1,200 rows, reducing 24 raw spellings to six cities. Canonicalized gender spellings to male/female; 30 blanks became unknown. Six ages outside 10–100 were set missing; together with 61 blank ages, 67 values were filled using the median age for that city via `groupby` and `transform`. Signup dates were parsed and stored as `YYYY-MM-DD`. The cleaned city counts are Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191 and Mumbai 184. Gender counts are female 610, male 560 and unknown 30; no ages are missing and all are 18–60.
- Orders: parsed `order_ts`, removed 119 exact duplicate rows, then kept the latest timestamp for each order ID, leaving 12,309 rows. This removes the 656 pending records; final status counts are 11,110 completed and 1,199 cancelled. Converted amounts to numeric and regenerated 120 unparseable amounts as `round(price_inr × quantity × (1 − discount_pct / 100), 2)`. Replaced all 106 negative delivery durations with missing values. Imputed missing durations only for completed orders using the median for that product category among completed orders; the result has zero missing completed durations. All 1,199 cancelled orders retain missing delivery duration because they were never delivered. Completed order amount totals INR 2,998,668.80.
- Clean customers, products and orders are saved under `data/clean/` and as `customers`, `products` and `orders` in `data/instamart.db`. `sql/queries.sql` contains seven runnable queries. The validation script asserts ten checkpoints and prints `ALL CHECKPOINTS PASSED`.

### SQL and pandas analysis

| Query | Result |
|---|---|
| Q1 — completed orders above INR 500, customers whose qualifying spend exceeds INR 8,000 | 30 customers |
| Q2 — completed revenue by city × category | 30 cells; grand total INR 2,998,668.80; largest cell Bengaluru / Electronics at INR 233,909.45 |
| Q3 — products priced above the catalogue average | 11 products |
| Q4 — customers without a completed order | 25 customers |
| Q5 — customer spend above the average customer total | 550 customers; average INR 2,552.06 |
| Q6 — city-wise RANK, DENSE_RANK and ROW_NUMBER | 8 rows have RANK different from ROW_NUMBER. Tied totals share a rank, so subsequent ranks skip positions while row numbers remain unique. |
| Q7 — daily running total | 183 days; final cumulative revenue INR 2,998,668.80 |

Q6 city leaders (completed spend): Bengaluru C0082, INR 12,856.20; Chennai C0930, INR 10,151.40; Delhi C0291, INR 13,742.60; Hyderabad C0743, INR 12,730.05; Mumbai C0693, INR 14,766.55; Pune C0426, INR 10,377.10. A pandas `pivot_table` with margins reproduces the SQL Q2 grand total, INR 2,998,668.80.

| City | Completed sum | Mean order | Orders |
|---|---:|---:|---:|
| Bengaluru | INR 523,643.40 | INR 276.47 | 1,894 |
| Chennai | INR 463,629.15 | INR 272.72 | 1,700 |
| Delhi | INR 502,829.80 | INR 243.62 | 2,064 |
| Hyderabad | INR 528,589.90 | INR 284.95 | 1,855 |
| Mumbai | INR 479,058.50 | INR 297.00 | 1,613 |
| Pune | INR 500,918.05 | INR 252.48 | 1,984 |

Completed order amount has mean INR 269.91, median INR 135.00, skewness 4.61, Q1 INR 64.80 and Q3 INR 297.00. The 1.5×IQR fences are INR −283.50 and INR 645.30; there are 0 lower-fence and 897 upper-fence outliers. Among order amount, discount, quantity and delivery duration, order amount and discount have the largest absolute correlation at −0.095 (order amount and quantity are +0.087); a correlation of this size does not establish causation.

The dataset describes six months of fictional grocery, dairy, snack, personal-care and small-electronics orders across six cities. Amounts are strongly right-skewed: the mean is twice the median, and high-value orders create 897 upper outliers. Negative and blank delivery times, duplicate orders, malformed amounts, mixed city/gender spellings and pending records were raw-data anomalies; after cleaning, every completed order has an amount and delivery duration. Discount has a small negative association with order amount, while quantity has a slightly smaller positive association; these weak correlations are descriptive and do not establish causal relationships.

## Part B

### RFM and customer segments

At the 2025-10-01 snapshot, RFM covers 1,175 customers: recency median 19 days (minimum 0, maximum 180), frequency mean 9.46 (maximum 33), and monetary mean INR 2,552.06 (maximum INR 14,766.55). K-means uses standardized RFM and `n_init=10`, `random_state=42`.

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1,820.8 | 0.465 |
| 3 | 1,268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

We choose K=4: it has the strongest silhouette and a clear elbow with four useful customer groups. Inertia always falls as K increases because adding clusters can only reduce or preserve within-cluster squared distance; K=8's lower inertia alone cannot distinguish useful structure from splitting customers into increasingly small groups.

| Segment ID | Name | Size | Mean recency (days) | Mean frequency | Mean monetary | Recommended action |
|---:|---|---:|---:|---:|---:|---|
| 0 | At-Risk Regulars | 436 | 28.86 | 3.77 | INR 1,057.62 | Send a low-friction replenishment reminder and a relevant offer. |
| 1 | Premium Champions — Premium | 138 | 24.12 | 6.77 | INR 7,129.70 | Offer priority support and curated premium bundles. |
| 2 | Steady Value | 419 | 8.77 | 19.40 | INR 3,410.02 | Encourage repeat baskets with loyalty rewards. |
| 3 | Occasional Essentials | 182 | 109.04 | 2.20 | INR 685.99 | Test a win-back message focused on essentials. |

Cluster IDs are arbitrary labels with no order. Segment 1 has the highest mean monetary value and is the Premium group. The two PCA components explain 0.6343 and 0.2180 of scaled RFM variance (0.8523 combined). `outputs/customer_segments.csv` includes each customer's RFM values, segment ID/name and Premium flag.

### Lapse model

The cutoff is 2025-09-01. The pre-cutoff cohort has 1,156 customers; 443 (38.32%) lapsed in September. A stratified 80/20 split gives 924 training and 232 test rows. Always predicting not-lapsed yields 61.6% accuracy and lapsed-class F1 of 0.

Features are pre-cutoff recency, frequency, monetary, mean discount, cancellation rate, average completed delivery minutes, favourite category, city, age, gender and tenure. No order on or after 2025-09-01 enters these features; Part B segment labels are not model features. Numeric scaling and k−1 one-hot encoding are inside the pipeline and fitted only on training rows. No encoder, scaler or resampler saw test rows, and code asserts no feature is missing. The fitted frequency scaler mean is 8.115 on training rows, versus 8.029 in the full cohort because the training split is a sample rather than the whole cohort.

| Model | Best parameters | Best CV F1 | Test confusion matrix (TN, FP / FN, TP) | Lapsed precision | Lapsed recall | Lapsed F1 |
|---|---|---:|---|---:|---:|---:|
| Logistic regression | C=100 | 0.651 | 111, 32 / 31, 58 | 0.644 | 0.652 | 0.648 |
| Random forest | depth=8, min_samples_leaf=10 | 0.687 | 111, 32 / 32, 57 | 0.640 | 0.640 | 0.640 |

The selected model is the random forest because it had the higher cross-validated F1 after the specified grid searches, while its test F1 remains above the 0.55 target. Missing a lapsing customer loses a chance to retain future orders; wrongly targeting a loyal customer spends an offer or outreach effort unnecessarily. The chosen model balances both on the held-out set, though the retention team can tune its contact threshold to favor recall if outreach capacity permits.

Oversampling duplicated only lapsed training rows until both training classes had 570 records each. On the unchanged test set, logistic recall increased from 0.652 to 0.854, precision moved from 0.644 to 0.589, and F1 rose from 0.648 to 0.697. Recall moved most, meaning the retention team catches more at-risk customers but sends more unnecessary outreach.

The five leading random forest features are monetary (0.2779), frequency (0.2768), recency (0.1113), average delivery duration (0.0617) and average discount (0.0532). The three largest logistic coefficients are frequency (negative, −1.6973: more prior orders associate with lower lapse risk), electronics as favourite category (negative, −1.1351: lower estimated risk in this cohort), and grocery as favourite category (negative, −0.4492: also lower estimated risk than the reference category). The persisted random forest is reloaded and produces exactly the same held-out predictions. The top 10 test-set customers marked Premium contain 1 who actually lapsed. That Premium flag comes from whole-window RFM including September, so the retention list is illustrative and not a leakage-free evaluation.

## Part C

The pipeline processes tickets in file order. Empty/whitespace tickets (4) and tickets over 400 estimated tokens (4; 734 tokens each) never reach the model. Before prompting, it redacts email, phone and order IDs in that order; across 164 tickets this replaced 55 emails, 91 phones and 132 order IDs (278 total), and the code asserts none of the three patterns remain. The user message is exactly the redacted text.

Both system prompts require a JSON-only category/reason response and list billing, delivery, technical and fraud. The few-shot prompt includes two labelled examples, including fraud. Transient errors are retried up to three times with exponential backoff and jitter; each full run resets the mock. There are 19 transient errors per full run, all recovered, recorded in `outputs/retry_log.txt`.

| Prompt | Classified | Accuracy on classified tickets | Response tokens | Tokens per classified ticket |
|---|---:|---:|---:|---:|
| Zero-shot | 147 (135 JSON, 12 keyword) | 126/147 = 85.7% | 8,776 | 59.70 |
| Few-shot | 147 (135 JSON, 12 keyword) | 147/147 = 100% | 12,520 | 85.17 |

Few-shot costs 3,744 more estimated tokens overall, or 25.47 more per classified ticket (about 42.7%); on this mock data it improves accuracy by 14.3 percentage points and avoids all 21 zero-shot misclassifications, so the added context is worthwhile when classification precision matters. For lower-risk high-volume routing, the zero-shot cost may be preferable with human review for uncertain cases.

Zero-shot confusion table (rows are true labels; columns are model labels):

| True \\ Predicted | billing | delivery | technical | fraud |
|---|---:|---:|---:|---:|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| technical | 10 | 0 | 29 | 0 |
| fraud | 11 | 0 | 0 | 23 |

JSON categories outside the allowed set are sent to human review. Non-JSON replies are accepted only when regex-extracted lowercase words contain exactly one allowed category; otherwise the ticket is flagged. Full-run statuses are 135 `classified_json`, 12 `classified_keyword` and 9 `flagged_invalid_output`, plus 4 `skipped_empty` and 4 `flagged_long`. The zero-shot budget demonstration uses 6,000 estimated tokens and stops 49 further tickets as `budget_exhausted`. Final few-shot routing uses a 100,000 token budget and writes 164 rows to `outputs/triage_results.csv`: empty, long, invalid and exhausted tickets go to `human_review` at NORMAL priority; valid fraud goes to `security_team` at HIGH; Premium customers go to `premium_desk` at HIGH; others go to `support_queue` at NORMAL. The segment name is blank if a ticket has no customer segment row.

## Part D

### Policy retrieval and answers

The five policy files become 15 chunks using 40 words and 10-word overlap: refund 4, delivery 3, escalation 3, payment 3 and privacy 2. `TfidfVectorizer()` fits the chunk texts; cosine similarity ranks the results. Term frequency measures how often a term appears in a chunk, while inverse document frequency downweights terms common across chunks. TF-IDF can miss equivalent phrasings that use different words because it does not model meaning or synonyms.

For `refund policy late delivery`, all three top chunks come from `refund_policy.txt`. On the 12 covered evaluation questions, the expected document appears in the top three for 12/12 and the case-sensitive expected fact appears in 12/12 answers. Both uncovered questions receive the required `I could not find` fallback (2/2). Repeating the evaluation with zero overlap yields 10/12 expected facts; the question “What delivery time makes a completed order eligible for a full refund?” fails when the relevant threshold sentence crosses a chunk boundary. The payment refund timing question also fails under zero overlap.

### Refund tool agent

The agent uses `get_order`, `search_policy`, `calculate_refund`, `issue_refund` and `escalate_to_human`. Only completed orders delivered after 45 minutes qualify for the full order amount; orders at 45 minutes or less receive zero, and amounts above INR 1,000 are escalated rather than automatically issued. The seven scenario outcomes are:

| Scenario | Status | Refund amount | Model calls | Tokens |
|---|---|---:|---:|---:|
| S1 | REFUND | INR 128.25 | 5 | 912 |
| S2 | REFUND | INR 135.00 | 5 | 912 |
| S3 | ESCALATE | INR 1,999.00 | 5 | 899 |
| S4 | DENIED | INR 0.00 | 4 | 617 |
| S5 | DENIED | INR 0.00 | 3 | 346 |
| S6 | NOT_FOUND | INR 0.00 | 2 | 78 |
| S7 | FALLBACK_MAX_ITERATIONS | INR 0.00 | 6 | 3,018 |

The S1 run with `token_limit=200` ends as `FALLBACK_TOKEN_LIMIT`. The S1 run with `time_limit_seconds=0` ends as `FALLBACK_TIMEOUT`. S7 completes six calls under the default 6,000 token limit, then ends with `FALLBACK_MAX_ITERATIONS`. Each fallback includes a human-handoff message and completed steps. The crash-and-resume S1 demonstration executes exactly these tools across both runs, without repeating a tool: `get_order` → `search_policy` → `calculate_refund` → `issue_refund`; it finishes as REFUND for INR 128.25. Checkpoints in `state/` retain the messages, token count, step number and executed tools.

ReAct measures 5 calls for S1, 3 for S5 and 2 for S6 because each observation can change the next action. A plan-and-execute version would usually use fewer calls for the normal path by asking for a step list once, but it would need a new call if an order is missing, cancelled or yields an unexpected observation; its plan can otherwise continue from assumptions that the tools disproved. ReAct spends more calls and tokens on normal cases but uses the observed database and policy results at each step. For refunds I choose ReAct because checking order status, delivery time and escalation amount before issuing money is more important than minimizing a few offline calls.

## Limitations and Next Steps

All records, policy language and outcomes are synthetic; the policy files are scenario material, not real company policy. TF-IDF and keyword rules do not understand paraphrases well, and the mock model does not represent production model behavior. Lapse labels cover one September window, and the random holdout is not a time-based production evaluation. The Premium flag uses whole-window RFM including September, so the retention list is illustrative and not a leakage-free evaluation. Before any real deployment, replace synthetic policies with approved policy documents, audit privacy and refund controls, evaluate on future time windows, monitor drift and human overrides, and set contact thresholds to match outreach capacity.