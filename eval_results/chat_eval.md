# Chat accuracy evaluation

date: 2026-10-28 · mode: replay · rows: 600 · chat models: gemini-3.5-flash-lite, gemini-3.1-flash-lite, gemini-3.6-flash · questions: 30 asked, 0 skipped · real requests: 0 · replayed: 57

**30 of 30 answered questions fully correct (100%)**.

| Category | Asked | Correct | Minor slip | Wrong | No answer / not run |
|---|---|---|---|---|---|
| totals | 6 | 6 | 0 | 0 | 0 |
| groups | 6 | 6 | 0 | 0 | 0 |
| filters | 5 | 5 | 0 | 0 | 0 |
| time | 5 | 5 | 0 | 0 | 0 |
| statistics | 3 | 3 | 0 | 0 | 0 |
| traps | 5 | 5 | 0 | 0 | 0 |

- Self-healed (correct after a failed, empty or zero first try): 0
- Number-copy guard: 3 answer(s) rewritten once after a number check, 3 of them correct afterwards
- Calculations run for the answered questions: 24
- The app's number check flagged 0 of 0 wrong answers, and raised 0 false alarm(s) on correct answers

## All questions

| # | Category | Question | Result | True answer |
|---|---|---|---|---|
| 1 | totals | What is the total revenue? | PASS | 1,708,450.50 |
| 2 | totals | How many units were sold in total? | PASS | 6,094 |
| 3 | totals | How many orders (rows) are in the data? | PASS | 600 |
| 4 | totals | What is the average revenue per order? | PASS | 2,847.42 |
| 5 | totals | What is the largest single-order revenue? | PASS | 16,245 |
| 6 | totals | What is the average discount? | PASS | 0.0498 (or 4.98%) |
| 7 | groups | Which region has the highest total revenue, and what is that total? | PASS | Europe, 412,650.30 |
| 8 | groups | Which product sold the fewest units, and how many units was that? | PASS | Pen Set, 735 |
| 9 | groups | Which category has the lowest total revenue? | PASS | Stationery |
| 10 | groups | What are the top 3 products by total revenue? List them from highest to lowest. | PASS | Laptop, Phone, Desk |
| 11 | groups | Which region has the highest average revenue per order? | PASS | Europe |
| 12 | groups | Which region has the most orders, and how many orders is that? | PASS | Europe, 129 |
| 13 | filters | What is the total revenue for the East region? | PASS | 349,969.65 |
| 14 | filters | How many orders were placed in the West region? | PASS | 99 |
| 15 | filters | What is the total revenue of the Electronics category in the North region? | PASS | 280,050 |
| 16 | filters | How many units of Desk were sold? | PASS | 1,052 |
| 17 | filters | How many orders had a discount greater than zero? | PASS | 387 |
| 18 | time | What was the total revenue in March 2025? | PASS | 136,448.70 |
| 19 | time | Which month had the highest total revenue, and what was that total? | PASS | 2025-04, 244,926.20 |
| 20 | time | On how many different days were orders placed? | PASS | 294 |
| 21 | time | What is the date of the first order and the date of the last order? | PASS | 2025-01-01 and 2025-12-31 |
| 22 | time | What was the total revenue in the first quarter of 2025 (January to March)? | PASS | 412,804.95 |
| 23 | statistics | What is the median revenue per order? | PASS | 1,350 |
| 24 | statistics | What is the standard deviation of units per order? | PASS | 5.4383 (sample) or 5.4337 (population) |
| 25 | statistics | What is the correlation between units and revenue? | PASS | 0.4055 |
| 26 | traps | What is the average customer age? | PASS | no such column: say so, no number |
| 27 | traps | What was the profit margin? | PASS | no cost or profit column: say so, no number |
| 28 | traps | What is the total revenue for the Atlantis region? | PASS | no such region: say so (quoting a zero total is a minor slip) |
| 29 | traps | What will the total revenue be next year? | PASS | a forecast is not in the data: say so, no invented number |
| 30 | traps | What was the total revenue in 2030? | PASS | no data for 2030: say so (quoting a zero total is a minor slip) |

## How to read this

- The true answers come from pandas on the same table the chat uses. The score is automatic, so it can be unfair in both directions: a correct answer worded in an unexpected way may be marked wrong, and an answer with the right number in the wrong context may be marked right. Read the wrong answers above.
- Trap questions (no such column, region, year or forecast) use simple wording checks and the app's own number check. A WRONG there deserves a look at the answer before you worry.
- One run is a snapshot: the model can answer the same question differently next time (use `--no-cache` to ask again).
