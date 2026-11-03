# Chat accuracy across several tables (Olist, demo set)

date: 2026-11-03 · mode: use · tables: 8 · analysed: order_items_flat · chat models: gemini-3.5-flash-lite, gemini-3.1-flash-lite, gemini-3.6-flash · questions: 14 asked, 0 skipped · real requests: 26 · replayed: 0

**13 of 14 answered questions fully correct (93%)**.

| Category | Asked | Correct | Minor slip | Wrong | No answer / not run |
|---|---|---|---|---|---|
| one table | 3 | 3 | 0 | 0 | 0 |
| lookup columns | 3 | 2 | 0 | 1 | 0 |
| other event table | 4 | 4 | 0 | 0 | 0 |
| two event tables | 2 | 2 | 0 | 0 | 0 |
| traps | 2 | 2 | 0 | 0 | 0 |

- Self-healed (correct after a failed, empty or zero first try): 0
- Number-copy guard: 0 answer(s) rewritten once after a number check, 0 of them correct afterwards
- Calculations run for the answered questions: 12
- The app's number check flagged 0 of 1 wrong answers, and raised 0 false alarm(s) on correct answers

## Wrong answers

**6. Which customer state placed the most orders, and how many?**
- true answer: SP, 3012 orders placed by customers there
- the AI said: 'The customer state that placed the most orders is SP (São Paulo), with a total of 2,971 orders.'
- expected 3,012; numbers in the answer: 2971

## All questions

| # | Category | Question | Result | True answer |
|---|---|---|---|---|
| 1 | one table | How many orders have the status 'canceled'? | PASS | 37 orders with order_status 'canceled' |
| 2 | one table | How many distinct sellers are there? | PASS | 965 distinct sellers (unique seller_id in sellers) |
| 3 | one table | How many orders were delivered later than their estimated delivery date? | PASS | 1043 orders delivered later than their estimated delivery date |
| 4 | lookup columns | Which product category (in English) appears in the most order items, and how many? | PASS | bed_bath_table, 975 order items |
| 5 | lookup columns | Which seller state shipped the most order items, and how many? | PASS | SP, 6238 order items shipped by sellers there |
| 6 | lookup columns | Which customer state placed the most orders, and how many? | FAIL | SP, 3012 orders placed by customers there |
| 7 | other event table | What is the total value recorded in order payments? | PASS | R$ 1,194,882.80 total (sum of order_payments.payment_value) |
| 8 | other event table | How many reviews have a score of 5? | PASS | 3992 reviews with review_score == 5 |
| 9 | other event table | What is the average review score across all reviews? | PASS | 3.894: mean of order_reviews.review_score |
| 10 | other event table | Which payment type is used the most, and how many times? | PASS | credit_card, used 5897 times in order_payments |
| 11 | two event tables | How many orders were paid by credit card and also received a review score of 1? | PASS | 919 orders paid by credit_card that also have a review_score of 1 |
| 12 | two event tables | What is the average review score for orders that contain a health_beauty item? | PASS | 4.115: mean review_score for orders containing a health_beauty item (530 such orders) |
| 13 | traps | What is the average customer age? | PASS | no age or birth-date column exists in any table |
| 14 | traps | How many orders were placed in 2025? | PASS | 0: the data runs from September 2016 to October 2018, so there is no 2025 |

## How to read this

- The true answers come from pandas on the same table the chat uses. The score is automatic, so it can be unfair in both directions: a correct answer worded in an unexpected way may be marked wrong, and an answer with the right number in the wrong context may be marked right. Read the wrong answers above.
- Trap questions (no such column, region, year or forecast) use simple wording checks and the app's own number check. A WRONG there deserves a look at the answer before you worry.
- One run is a snapshot: the model can answer the same question differently next time (use `--no-cache` to ask again).
