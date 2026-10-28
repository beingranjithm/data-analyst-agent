# Chat accuracy across several tables (Formula 1)

date: 2026-10-28 · mode: use · tables: 14 · analysed: results_flat · chat models: gemini-3.5-flash-lite, gemini-3.1-flash-lite, gemini-3.6-flash · questions: 18 asked, 0 skipped · real requests: 4 · replayed: 32

**18 of 18 answered questions fully correct (100%)**.

| Category | Asked | Correct | Minor slip | Wrong | No answer / not run |
|---|---|---|---|---|---|
| one table | 2 | 2 | 0 | 0 | 0 |
| lookup columns | 9 | 9 | 0 | 0 | 0 |
| other event table | 4 | 4 | 0 | 0 | 0 |
| two event tables | 1 | 1 | 0 | 0 | 0 |
| traps | 2 | 2 | 0 | 0 | 0 |

- Self-healed (correct after a failed, empty or zero first try): 2
- Number-copy guard: 0 answer(s) rewritten once after a number check, 0 of them correct afterwards
- Calculations run for the answered questions: 18
- The app's number check flagged 0 of 0 wrong answers, and raised 0 false alarm(s) on correct answers

## All questions

| # | Category | Question | Result | True answer |
|---|---|---|---|---|
| 1 | one table | How many races were held in 2019? | PASS | 21 races in 2019 (count of rows in races) |
| 2 | one table | How many drivers are Brazilian? | PASS | 32 drivers with nationality Brazilian |
| 3 | lookup columns | Which driver has won the most races, and how many? | PASS | Lewis Hamilton, 105 wins |
| 4 | lookup columns | Which constructor has won the most races, and how many? | PASS | Ferrari, 249 wins |
| 5 | lookup columns | How many different drivers have ever won a race? | PASS | 115 different drivers have won a race |
| 6 | lookup columns | How many races did Michael Schumacher win? | PASS | Michael Schumacher: 91 wins |
| 7 | lookup columns | What is the total number of points Lewis Hamilton has scored? | PASS | 4820.5 points (sum of results.points for Hamilton) |
| 8 | lookup columns | Which circuit has hosted the most races, and how many? | PASS | Autodromo Nazionale di Monza, 74 races |
| 9 | lookup columns | Which country has hosted the most races, and how many? | PASS | Italy, 107 races |
| 10 | lookup columns | Which driver scored the most points in 2019, and how many? | PASS | Hamilton, 413 points in 2019 |
| 11 | lookup columns | How many different constructors took part in 2020? | PASS | 10 different constructors scored/entered a result in 2020 |
| 12 | other event table | How many pit stops did Max Verstappen make in 2021? | PASS | 43 pit-stop rows for Verstappen in 2021 |
| 13 | other event table | How many laps did Lewis Hamilton drive in 2019 according to the lap times? | PASS | 1262 lap rows for Hamilton in 2019 |
| 14 | other event table | What was Lewis Hamilton's average lap time in milliseconds in 2019? | PASS | 91,899.3 ms: mean of lap_times.milliseconds for Hamilton in 2019 |
| 15 | other event table | Which driver set the fastest lap in the lap times data? | PASS | Russell (55,404 ms is the shortest lap in lap_times) |
| 16 | two event tables | How many pit stops did the winner of the 2019 Monaco Grand Prix make, and who was it? | PASS | winner Hamilton, 1 pit stops in the 2019 Monaco Grand Prix |
| 17 | traps | What was the average attendance at each race? | PASS | no attendance data exists in any table |
| 18 | traps | Who won the 2030 world championship? | PASS | the data ends in 2024; there is no 2030 season |

## How to read this

- The true answers come from pandas on the same table the chat uses. The score is automatic, so it can be unfair in both directions: a correct answer worded in an unexpected way may be marked wrong, and an answer with the right number in the wrong context may be marked right. Read the wrong answers above.
- Trap questions (no such column, region, year or forecast) use simple wording checks and the app's own number check. A WRONG there deserves a look at the answer before you worry.
- One run is a snapshot: the model can answer the same question differently next time (use `--no-cache` to ask again).
