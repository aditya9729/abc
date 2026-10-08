# Passed with warnings: 1 finding \(1 minor\)

Change e62f0504198e against 192ae8b, 3 files, +588 -6

Blast radius: risk high \(26 symbols touched, 23 callers in 4 files\)

Counts: 1 finding \(1 minor\), 36 scanner candidates dropped

## Findings \(1\)

### 1. Minor bug: Misleading error for a malformed or uppercase pin

- **Where:** abc\_bench/dashboard.py:257-261
- **Problem:** A pin that is present but not lowercase hex fails the regex and raises the 'must be supplied together' message. Uppercase digests from tools like PowerShell Get-FileHash therefore produce an error that describes a different fault.
- **Why it matters:** The operator sees a message saying an argument is missing when both arguments were given. They cannot tell the pin format is the problem.
- **Fix:** Check the regex in a separate branch with its own message, or lowercase the pin before matching. Keep the hash comparison on the normalised value.
- **Source:** the reviewer

## Dropped scanner candidates \(36\)

### c1 at tests/bench/test\_dashboard\_vla.py:185: The URL is a fixed loopback base built from the test server's own port, not user input. \(see tests/bench/test\_dashboard\_vla.py:183\)

- **Source:** semgrep:python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected

### c2 at tests/bench/test\_dashboard\_vla.py:187: The URL is a fixed loopback base built from the test server's own port, not user input. \(see tests/bench/test\_dashboard\_vla.py:183\)

- **Source:** semgrep:python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected

### c3 at tests/bench/test\_dashboard\_vla.py:190: The URL is a fixed loopback base built from the test server's own port, not user input. \(see tests/bench/test\_dashboard\_vla.py:183\)

- **Source:** semgrep:python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected

### c4 at tests/bench/test\_dashboard\_vla.py:185: The scheme is a hardcoded http loopback address in a test, so no file or custom scheme can be reached. \(see tests/bench/test\_dashboard\_vla.py:183\)

- **Source:** bandit:B310

### c5 at tests/bench/test\_dashboard\_vla.py:187: The scheme is a hardcoded http loopback address in a test, so no file or custom scheme can be reached. \(see tests/bench/test\_dashboard\_vla.py:183\)

- **Source:** bandit:B310

### c6 at tests/bench/test\_dashboard\_vla.py:190: The scheme is a hardcoded http loopback address in a test, so no file or custom scheme can be reached. \(see tests/bench/test\_dashboard\_vla.py:183\)

- **Source:** bandit:B310

### c7 at tests/bench/test\_dashboard\_vla.py:75: Assert statements in a pytest test module are the intended test mechanism. \(see tests/bench/test\_dashboard\_vla.py:75\)

- **Source:** bandit:B101

### c8 at tests/bench/test\_dashboard\_vla.py:76: Assert statements in a pytest test module are the intended test mechanism. \(see tests/bench/test\_dashboard\_vla.py:76\)

- **Source:** bandit:B101

### c9 at tests/bench/test\_dashboard\_vla.py:77: Assert statements in a pytest test module are the intended test mechanism. \(see tests/bench/test\_dashboard\_vla.py:77\)

- **Source:** bandit:B101

### c10 at tests/bench/test\_dashboard\_vla.py:78: Assert statements in a pytest test module are the intended test mechanism. \(see tests/bench/test\_dashboard\_vla.py:78\)

- **Source:** bandit:B101

### c11 at tests/bench/test\_dashboard\_vla.py:79: Assert statements in a pytest test module are the intended test mechanism. \(see tests/bench/test\_dashboard\_vla.py:79\)

- **Source:** bandit:B101

### c12 at tests/bench/test\_dashboard\_vla.py:80: Assert statements in a pytest test module are the intended test mechanism. \(see tests/bench/test\_dashboard\_vla.py:80\)

- **Source:** bandit:B101

### c13 at tests/bench/test\_dashboard\_vla.py:84: Assert statements in a pytest test module are the intended test mechanism. \(see tests/bench/test\_dashboard\_vla.py:84\)

- **Source:** bandit:B101

### c14 at tests/bench/test\_dashboard\_vla.py:86: Assert statements in a pytest test module are the intended test mechanism. \(see tests/bench/test\_dashboard\_vla.py:86\)

- **Source:** bandit:B101

### c15 at tests/bench/test\_dashboard\_vla.py:91: Assert statements in a pytest test module are the intended test mechanism. \(see tests/bench/test\_dashboard\_vla.py:91\)

- **Source:** bandit:B101

### c16 at tests/bench/test\_dashboard\_vla.py:93: Assert statements in a pytest test module are the intended test mechanism. \(see tests/bench/test\_dashboard\_vla.py:93\)

- **Source:** bandit:B101

### c17 at tests/bench/test\_dashboard\_vla.py:99: Assert statements in a pytest test module are the intended test mechanism. \(see tests/bench/test\_dashboard\_vla.py:99\)

- **Source:** bandit:B101

### c18 at tests/bench/test\_dashboard\_vla.py:100: Assert statements in a pytest test module are the intended test mechanism. \(see tests/bench/test\_dashboard\_vla.py:100\)

- **Source:** bandit:B101

### c19 at tests/bench/test\_dashboard\_vla.py:101: Assert statements in a pytest test module are the intended test mechanism. \(see tests/bench/test\_dashboard\_vla.py:101\)

- **Source:** bandit:B101

### c20 at tests/bench/test\_dashboard\_vla.py:130: Assert statements in a pytest test module are the intended test mechanism. \(see tests/bench/test\_dashboard\_vla.py:130\)

- **Source:** bandit:B101

### c21 at tests/bench/test\_dashboard\_vla.py:131: Assert statements in a pytest test module are the intended test mechanism. \(see tests/bench/test\_dashboard\_vla.py:131\)

- **Source:** bandit:B101

### c22 at tests/bench/test\_dashboard\_vla.py:145: Assert statements in a pytest test module are the intended test mechanism. \(see tests/bench/test\_dashboard\_vla.py:145\)

- **Source:** bandit:B101

### c23 at tests/bench/test\_dashboard\_vla.py:155: Assert statements in a pytest test module are the intended test mechanism. \(see tests/bench/test\_dashboard\_vla.py:155\)

- **Source:** bandit:B101

### c24 at tests/bench/test\_dashboard\_vla.py:169: Assert statements in a pytest test module are the intended test mechanism. \(see tests/bench/test\_dashboard\_vla.py:169\)

- **Source:** bandit:B101

### c25 at tests/bench/test\_dashboard\_vla.py:186: Assert statements in a pytest test module are the intended test mechanism. \(see tests/bench/test\_dashboard\_vla.py:186\)

- **Source:** bandit:B101

### c26 at tests/bench/test\_dashboard\_vla.py:188: Assert statements in a pytest test module are the intended test mechanism. \(see tests/bench/test\_dashboard\_vla.py:188\)

- **Source:** bandit:B101

### c27 at tests/bench/test\_dashboard\_vla.py:191: Assert statements in a pytest test module are the intended test mechanism. \(see tests/bench/test\_dashboard\_vla.py:191\)

- **Source:** bandit:B101

### c28 at tests/bench/test\_dashboard\_vla.py:200: Assert statements in a pytest test module are the intended test mechanism. \(see tests/bench/test\_dashboard\_vla.py:200\)

- **Source:** bandit:B101

### c29 at tests/bench/test\_dashboard\_vla.py:201: Assert statements in a pytest test module are the intended test mechanism. \(see tests/bench/test\_dashboard\_vla.py:201\)

- **Source:** bandit:B101

### c30 at tests/bench/test\_dashboard\_vla.py:202: Assert statements in a pytest test module are the intended test mechanism. \(see tests/bench/test\_dashboard\_vla.py:202\)

- **Source:** bandit:B101

### c31 at tests/bench/test\_dashboard\_vla.py:203: Assert statements in a pytest test module are the intended test mechanism. \(see tests/bench/test\_dashboard\_vla.py:203\)

- **Source:** bandit:B101

### c32 at tests/bench/test\_dashboard\_vla.py:204: Assert statements in a pytest test module are the intended test mechanism. \(see tests/bench/test\_dashboard\_vla.py:204\)

- **Source:** bandit:B101

### c33 at tests/bench/test\_dashboard\_vla.py:213: Assert statements in a pytest test module are the intended test mechanism. \(see tests/bench/test\_dashboard\_vla.py:213\)

- **Source:** bandit:B101

### c34 at tests/bench/test\_dashboard\_vla.py:221: Assert statements in a pytest test module are the intended test mechanism. \(see tests/bench/test\_dashboard\_vla.py:221\)

- **Source:** bandit:B101

### c35 at tests/bench/test\_dashboard\_vla.py:230: Assert statements in a pytest test module are the intended test mechanism. \(see tests/bench/test\_dashboard\_vla.py:230\)

- **Source:** bandit:B101

### c36 at tests/bench/test\_dashboard\_vla.py:231: Assert statements in a pytest test module are the intended test mechanism. \(see tests/bench/test\_dashboard\_vla.py:231\)

- **Source:** bandit:B101

## Coverage

- **Files read:** abc\_bench/dashboard.py, abc\_bench/vla\_comparison.py, tests/bench/test\_dashboard.py, tests/bench/test\_paired\_review.py
- **Files not read:** docs/bench/dashboard.md, tests/bench/test\_dashboard\_vla.py
- **Changed ranges given to the reviewer:** 16 of 16
Scanners: 4 scanners ran, 9 had nothing to check

Reviewer: claude 2.1.294, 122 s, 12 turns, 282,994 tokens in, 9,749 out, $1.56

Made by Qodex: review on every pull request at https://qodex.ai
