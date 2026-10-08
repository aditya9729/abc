# Passed: no findings

Change df1d1546ef18 against 4fb7748, 2 files, +12 -6

Blast radius: risk high \(2 symbols touched, 16 callers in 3 files\)

Counts: no findings, 4 scanner candidates dropped

## Findings \(0\)

No findings on the changed lines.

## Dropped scanner candidates \(4\)

### c1 at tests/bench/test\_dashboard\_vla.py:87: The assert is a pytest test assertion, which is the expected idiom in a test module. \(see tests/bench/test\_dashboard\_vla.py:87\)

- **Source:** bandit:B101

### c2 at tests/bench/test\_dashboard\_vla.py:90: The assert is a pytest test assertion, which is the expected idiom in a test module. \(see tests/bench/test\_dashboard\_vla.py:90\)

- **Source:** bandit:B101

### c3 at tests/bench/test\_dashboard\_vla.py:91: The assert is a pytest test assertion, which is the expected idiom in a test module. \(see tests/bench/test\_dashboard\_vla.py:91\)

- **Source:** bandit:B101

### c4 at tests/bench/test\_dashboard\_vla.py:92: The assert is a pytest test assertion, which is the expected idiom in a test module. \(see tests/bench/test\_dashboard\_vla.py:92\)

- **Source:** bandit:B101

## Coverage

- **Files read:** abc\_bench/dashboard.py
- **Files not read:** tests/bench/test\_dashboard\_vla.py
- **Changed ranges given to the reviewer:** 4 of 4
Scanners: 4 scanners ran, 9 had nothing to check

Reviewer: claude 2.1.294, 25 s, 3 turns, 39,901 tokens in, 836 out, $0.22

Made by Qodex: review on every pull request at https://qodex.ai
