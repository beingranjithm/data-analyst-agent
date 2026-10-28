# Chat accuracy across several tables (Formula 1, demo set)

date: 2026-10-28 · mode: replay · tables: 14 · analysed: results_flat · chat models: gemini-3.5-flash-lite, gemini-3.1-flash-lite, gemini-3.6-flash · questions: 11 asked, 0 skipped · real requests: 0 · replayed: 26

**11 of 11 answered questions fully correct (100%)**.

| Category | Asked | Correct | Minor slip | Wrong | No answer / not run |
|---|---|---|---|---|---|
| one table | 1 | 1 | 0 | 0 | 0 |
| lookup columns | 4 | 4 | 0 | 0 | 0 |
| other event table | 3 | 3 | 0 | 0 | 0 |
| two event tables | 1 | 1 | 0 | 0 | 0 |
| traps | 2 | 2 | 0 | 0 | 0 |

- Self-healed (correct after a failed, empty or zero first try): 4
- Number-copy guard: 0 answer(s) rewritten once after a number check, 0 of them correct afterwards
- Calculations run for the answered questions: 15
- The app's number check flagged 0 of 0 wrong answers, and raised 0 false alarm(s) on correct answers

## All questions

| # | Category | Question | Result | True answer |
|---|---|---|---|---|
| 1 | one table | How many races were held in 2022? | PASS | 22 races in 2022 |
| 2 | lookup columns | Which driver won the most races in 2021, and how many? | PASS | Verstappen, 10 wins in 2021 |
| 3 | lookup columns | Which constructor won the most races in 2022, and how many? | PASS | Red Bull, 17 wins in 2022 |
| 4 | lookup columns | How many different drivers took part in 2021? | PASS | 21 different drivers in 2021 |
| 5 | lookup columns | How many different countries hosted a race in 2022? | PASS | 20 different countries hosted a race in 2022 |
| 6 | other event table | How many pit stops did Max Verstappen make in 2022? | PASS | 47 pit-stop rows for Verstappen in 2022 |
| 7 | other event table | What was Lewis Hamilton's average lap time in milliseconds in 2021? | PASS | 96,207.2 ms: mean lap time of Hamilton in 2021 |
| 8 | other event table | Which driver set the fastest lap in the lap times data? | PASS | Verstappen (66,200 ms is the shortest lap in lap_times) |
| 9 | two event tables | How many pit stops did the winner of the 2022 Monaco Grand Prix make, and who was it? | PASS | winner Pérez, 3 pit stops in the 2022 Monaco Grand Prix |
| 10 | traps | What was the average attendance at each race? | PASS | no attendance data exists in any table |
| 11 | traps | Who won the 2030 world championship? | PASS | the data ends in 2024; there is no 2030 season |

## How to read this

- The true answers come from pandas on the same table the chat uses. The score is automatic, so it can be unfair in both directions: a correct answer worded in an unexpected way may be marked wrong, and an answer with the right number in the wrong context may be marked right. Read the wrong answers above.
- Trap questions (no such column, region, year or forecast) use simple wording checks and the app's own number check. A WRONG there deserves a look at the answer before you worry.
- One run is a snapshot: the model can answer the same question differently next time (use `--no-cache` to ask again).
