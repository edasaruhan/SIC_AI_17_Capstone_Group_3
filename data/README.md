# Data: IBM AML HI-Small

ARGUS uses the IBM AML HI-Small transaction file and its companion accounts table.
The data is synthetic, distributed under a separate dataset license and kept as a
local input rather than repository content.

## Required placement

From the repository root, place the files exactly here:

```text
data/raw/HI-Small_Trans.csv
data/raw/HI-Small_accounts.csv
```

The configured transaction path is `paths.raw_data` in `configs/base.yaml`. The
accounts path is the companion file beside it. Do not edit the raw files in place.

The following paths are ignored by Git:

```text
data/raw/*
data/interim/*
data/processed/*
```

Only `.gitkeep` placeholders are eligible for version control. Do not use
`git add -f` for a dataset or generated table.

## Provenance and integrity

IBM Research AML-Data HI-Small is the primary experimental dataset. PaySim is
outside the final experimental scope.

Upstream context:

- [IBM AML-Data repository](https://github.com/IBM/AML-Data)
- [IBM Research publication: Realistic Synthetic Financial Transactions for
  Anti-Money Laundering Models](https://research.ibm.com/publications/realistic-synthetic-financial-transactions-for-anti-money-laundering-models)

These links establish project provenance; the local hashes below define the exact
snapshot audited here. Do not assume a later upstream archive is byte-identical.

The following values come from a read-only full-file audit and are not inferred
from a quick sample.

| File | Bytes | Data rows | SHA-256 |
| --- | ---: | ---: | --- |
| `HI-Small_Trans.csv` | 475,664,283 | 5,078,345 | `b19d39f515523373f991b689c07e11e7b0b95c17a2c27a87d91584ae16c5b040` |
| `HI-Small_accounts.csv` | 34,053,187 | 518,581 | `786808526e33cfc441212dd6fccda7edfc24172149bed59c6ef59b186836b014` |

Both files are 7-bit ASCII and therefore valid UTF-8, have no BOM, use comma
delimiters, contain no quote bytes, and use LF line endings including the final
line.

Recheck local copies from PowerShell:

```powershell
Get-Item -LiteralPath '.\data\raw\HI-Small_Trans.csv', '.\data\raw\HI-Small_accounts.csv' |
  Select-Object Name, Length
Get-FileHash -Algorithm SHA256 -LiteralPath '.\data\raw\HI-Small_Trans.csv', '.\data\raw\HI-Small_accounts.csv'
```

A changed hash means it is a different source snapshot and its results must be
versioned separately.

## Raw transaction schema

The physical header contains two columns both named `Account`:

```text
Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering
```

| Position | Raw field | Canonical field | Safe type | Meaning / rule |
| ---: | --- | --- | --- | --- |
| 1 | `Timestamp` | `timestamp` | datetime | Parse exactly as `%Y/%m/%d %H:%M` |
| 2 | `From Bank` | `from_bank` | string | Zero-padded sender bank ID |
| 3 | first `Account` | `from_account` | string | Sender account ID |
| 4 | `To Bank` | `to_bank` | string | Zero-padded receiver bank ID |
| 5 | second `Account` | `to_account` | string | Receiver account ID |
| 6 | `Amount Received` | `amount_received` | float64/decimal | Receiver-side positive amount |
| 7 | `Receiving Currency` | `receiving_currency` | string/category | Currency of received amount |
| 8 | `Amount Paid` | `amount_paid` | float64/decimal | Sender-side positive amount |
| 9 | `Payment Currency` | `payment_currency` | string/category | Currency of paid amount |
| 10 | `Payment Format` | `payment_format` | string/category | Transfer/payment channel |
| 11 | `Is Laundering` | `is_laundering` | uint8/bool | Synthetic transaction label, `{0,1}` |

Some readers, including pandas, disambiguate the second raw `Account` as
`Account.1`. The loader must map by the validated IBM layout and must never allow
the receiving account to overwrite the sender account.

## Raw accounts schema

```text
Bank Name,Bank ID,Account Number,Entity ID,Entity Name
```

| Raw field | Canonical field | Safe type | Meaning / rule |
| --- | --- | --- | --- |
| `Bank Name` | `bank_name` | string | Synthetic institution name |
| `Bank ID` | `bank_id` | string | Unpadded decimal bank identifier |
| `Account Number` | `account_number` | string | Nine-character uppercase hexadecimal-like identifier |
| `Entity ID` | `entity_id` | string | Nine-character uppercase hexadecimal-like entity identifier |
| `Entity Name` | `entity_name` | string | Synthetic entity name |

## Identity and referential-integrity rule

Bank IDs are represented differently across the two files:

- every transaction-side bank ID is zero-prefixed;
- accounts-side `Bank ID` values are unpadded.

A raw string join therefore failed for every sender and every receiver row in the
full audit. Canonicalize a digit-only bank ID with:

```text
normalized_bank_id = bank_id with leading zeroes removed; all-zero input -> "0"
node_id = normalized_bank_id + "::" + uppercase(account_id)
```

After this normalization, the full audit found zero unmatched sender rows and zero
unmatched receiver rows. It found 518,581 unique composite account keys.

`Account Number` alone is not globally unique: 518,581 account rows contain
518,573 unique account-number strings. Eight distinct account numbers each occur
twice in different composite identities. Never merge nodes on account number alone.

The transaction graph contains 515,088 unique normalized bank-account nodes. Every
transaction node exists in the accounts table; 3,493 valid account rows are not
used as a transaction endpoint.

## Full-file quality facts

### Transactions

- 5,078,345/5,078,345 rows have exactly 11 fields.
- Empty, whitespace-only, and surrounding-whitespace field counts are zero.
- Timestamp range is `2022/09/01 00:00`–`2022/09/18 16:18`; invalid timestamps: 0.
- Labels: `0 = 5,073,168`, `1 = 5,177`; other labels: 0.
- Both amount columns are finite and strictly positive in every row.
- Exact duplicate occurrences after the first copy: 9 in 9 groups; each group has
  two byte-identical rows and label `0`.
- There are 15 observed currency values and 7 payment formats.

### Accounts

- 518,581/518,581 rows have exactly 5 fields.
- Empty, whitespace-only, and surrounding-whitespace field counts are zero.
- Exact duplicate rows: 0.
- Duplicate canonical composite IDs: 0.

## Duplicate policy

Do not automatically discard a row merely because sender, receiver, amount, or
timestamp repeats. Repeated edges carry behavioral information. The pipeline must:

1. preserve a stable raw row identity;
2. report exact full-row duplicates separately from repeated interactions;
3. keep the default handling decision explicit in the run manifest;
4. avoid changing label counts silently.

## Full audit and execution scopes

The audit streams every row and applies strict timestamp parsing, finite-number
checks, duplicate counters and composite-key membership checks. It writes the
full-source evidence to `artifacts/full/raw_data_audit.json`:

```powershell
.\.venv\Scripts\python.exe scripts/audit_raw_data.py
```

`configs/quick.yaml` limits feature and EDA work to the first 10,000
transactions in chronological source order. The verified quick pipeline generated
52-column features and EDA under `artifacts/quick/`, including 13 PNG figures and
12 CSV EDA tables. Those sampled outputs cannot be used to restate full-file facts.

The out-of-core full pipeline processed all 5,078,345 rows without feature
sampling. It wrote 52-column feature and chronological split
Parquet outputs under `artifacts/full/` in 304.93406899999536 seconds. DuckDB used a
bounded `2GB` limit and one thread. Full-exact EDA tables cover every row; plots use
a target-independent deterministic 100,000-row edge sample and NetworkX is capped
at 50,000 edges. Those sampled graph values are descriptive, not population facts.

## Data handling

- Retain the original files read-only and derive interim/processed files elsewhere.
- Record source hash, config, seed, sampling scope, and row counts per run.
- Avoid emitting entity names in aggregate logs or public screenshots.
- Treat labels as synthetic ground truth for research, not proof about real people.
- Review all suspicious-network evidence manually before any downstream decision.
