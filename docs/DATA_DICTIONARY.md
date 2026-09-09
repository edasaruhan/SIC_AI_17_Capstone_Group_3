# ARGUS AI Data Dictionary

**Scope:** Sprint 1 raw, canonical, and engineered data contracts  
**Runtime availability:** `PASS` — verified 10,000-row quick artifact  
**Full-file audit facts:** independently verified and persisted by the project

CSV files contain text, not intrinsic data types. “Type” in this document means
the safe canonical type after validation, not an assumption made by a CSV reader.

## 1. Raw transaction fields

The raw header has two fields both spelled `Account`. Their position determines
sender versus receiver; pandas commonly exposes the second as `Account.1`.

| Pos. | Raw field | Canonical field | Canonical type | Validation and meaning |
| ---: | --- | --- | --- | --- |
| 1 | `Timestamp` | `timestamp` | `datetime64[ns]` | Parse `%Y/%m/%d %H:%M`; source values are timezone-naive |
| 2 | `From Bank` | `from_bank` | string | Decimal sender bank ID; normalize leading zeroes for identity |
| 3 | first `Account` | `from_account` | string | Sender account; trim and uppercase for identity |
| 4 | `To Bank` | `to_bank` | string | Decimal receiver bank ID; normalize leading zeroes for identity |
| 5 | second `Account` / `Account.1` | `to_account` | string | Receiver account; trim and uppercase for identity |
| 6 | `Amount Received` | `amount_received` | float64 | Finite, non-negative receiver-side amount |
| 7 | `Receiving Currency` | `receiving_currency` | string/category | Currency applicable to `amount_received` |
| 8 | `Amount Paid` | `amount_paid` | float64 | Finite, non-negative sender-side amount |
| 9 | `Payment Currency` | `payment_currency` | string/category | Currency applicable to `amount_paid` |
| 10 | `Payment Format` | `payment_format` | string/category | Payment channel |
| 11 | `Is Laundering` | `is_laundering` | int8/bool | Synthetic transaction-level target; only `0` or `1` |

### Observed categorical domains

The full-file audit observed the same 15 values in both currency columns:

`Australian Dollar`, `Bitcoin`, `Brazil Real`, `Canadian Dollar`, `Euro`,
`Mexican Peso`, `Ruble`, `Rupee`, `Saudi Riyal`, `Shekel`, `Swiss Franc`,
`UK Pound`, `US Dollar`, `Yen`, and `Yuan`.

It observed seven payment formats: `ACH`, `Bitcoin`, `Cash`, `Cheque`,
`Credit Card`, `Reinvestment`, and `Wire`.

These are observed values, not a rule that permits silently coercing an unknown
future value. A changed domain must be reported in validation output.

## 2. Raw accounts fields

| Raw field | Canonical field | Canonical type | Validation and meaning |
| --- | --- | --- | --- |
| `Bank Name` | `bank_name` | string | Synthetic institution name; non-empty |
| `Bank ID` | `bank_id` | string | Decimal, unpadded ID; normalize before joining |
| `Account Number` | `account_number` | string | Non-empty account ID; observed as 9-char uppercase hexadecimal text |
| `Entity ID` | `entity_id` | string | Non-empty synthetic entity ID; observed as 9-char uppercase hexadecimal text |
| `Entity Name` | `entity_name` | string | Synthetic entity display name; avoid aggregate-log disclosure |

Account-table implementations may retain `raw_bank_id` alongside canonical
`bank_id` so source representation remains auditable.

## 3. Provenance and identity fields

| Field | Type | Contract |
| --- | --- | --- |
| `source_row_number` | int64 | Physical CSV line number, normally 2-based after the header; immutable provenance |
| `from_node_id` | string | `normalized(from_bank)::UPPERCASE(from_account)` |
| `to_node_id` | string | `normalized(to_bank)::UPPERCASE(to_account)` |
| accounts `node_id` | string | `normalized(bank_id)::UPPERCASE(account_number)` |
| `transaction_id` | string | `<source_filename>:row-<source_row_number>`, for example `HI-Small_Trans.csv:row-4744` |
| `partition` | category | `train`, `validation`, or `test`; emitted in `split_manifest.csv` |

The canonical bank normalizer accepts decimal text, removes leading zeroes, and
maps an all-zero identifier to `0`. Blank or non-decimal bank IDs fail validation.

Account number alone is not a key. The audited accounts table contains 518,581
rows but 518,573 unique account-number strings. The composite canonical node key is
unique for all 518,581 rows.

## 4. Transaction-local engineered fields

These features depend only on the current transaction and therefore do not require
historical state.

| Field | Type | Definition |
| --- | --- | --- |
| `log_amount_paid` | float64 | `log1p(amount_paid)` |
| `log_amount_received` | float64 | `log1p(amount_received)` |
| `same_bank` | int8 | 1 when canonical sender and receiver bank IDs match |
| `currency_match` | int8 | 1 when normalized payment and receiving currency match |
| `amount_difference_same_currency` | float64 nullable | `amount_paid - amount_received` only when currencies match; otherwise null |
| `amount_ratio_same_currency` | float64 nullable | `amount_paid / amount_received` only for matching currency and nonzero denominator; otherwise null |

No exchange-rate table exists in Sprint 1. Comparing amounts across currencies as
if they shared a unit would fabricate a financial relationship.

## 5. Calendar and event-gap fields

| Field | Type | Definition |
| --- | --- | --- |
| `hour` | int8 | Event hour, 0–23 |
| `day_of_week` | int8 | Monday=0 through Sunday=6 |
| `is_weekend` | int8 | 1 for Saturday or Sunday |
| `sender_seconds_since_previous` | float64 nullable | Seconds since sender’s latest event with timestamp strictly below `t` |
| `receiver_seconds_since_previous` | float64 nullable | Seconds since receiver’s latest event with timestamp strictly below `t` |

The first event for an entity has a null previous-event gap. Transactions sharing
timestamp `t` must receive the same prior state and must not become one another’s
“previous” event.

## 6. Strictly-prior history fields

`{window}` is configured in `features.history_windows`; Sprint 1 configuration uses
`1h`, `24h`, and `7d`. Generated suffixes are `1h`, `24h`, and `7d`.

| Field | Type | State visible immediately before `t` |
| --- | --- | --- |
| `sender_previous_transaction_count` | int64 | All earlier sender transactions |
| `sender_previous_outgoing_amount` | float64 | Sum of earlier `amount_paid` for sender |
| `sender_previous_unique_counterparties` | int64 | Distinct receivers first seen before `t` |
| `sender_burst_count_{window}` | int64 | Earlier sender transaction count inside `(t-window, t)` |
| `sender_rolling_outgoing_amount_{window}` | float64 | Earlier sender paid volume inside `(t-window, t)` |
| `receiver_previous_transaction_count` | int64 | All earlier receiver transactions |
| `receiver_previous_incoming_amount` | float64 | Sum of earlier `amount_received` for receiver |
| `receiver_previous_unique_counterparties` | int64 | Distinct senders first seen before `t` |
| `receiver_burst_count_{window}` | int64 | Earlier receiver transaction count inside `(t-window, t)` |
| `receiver_rolling_incoming_amount_{window}` | float64 | Earlier receiver received volume inside `(t-window, t)` |

All counts and volumes are zero when no eligible history exists. “Earlier” means
strict timestamp comparison, not earlier row position.

## 7. Directed graph-history fields

| Field | Type | State visible immediately before `t` |
| --- | --- | --- |
| `sender_prior_fan_out_degree` | int64 | Distinct receivers previously reached by sender |
| `sender_prior_fan_in_degree` | int64 | Distinct senders that previously paid the sender node |
| `receiver_prior_fan_out_degree` | int64 | Distinct receivers previously reached by receiver node |
| `receiver_prior_fan_in_degree` | int64 | Distinct senders that previously paid the receiver node |
| `pair_previous_transfer_count` | int64 | Earlier directed transfers for the exact sender→receiver pair |

Direction matters. Reversing an edge changes fan-in/fan-out meaning. Repeated
transfers remain separate events; a degree changes only on first prior observation
of a neighbor, while the pair count includes every eligible prior multiedge.

## 8. Split metadata fields

`artifacts/quick/metadata/split_metadata.json` was generated and verified with:

| Field | Verified value |
| --- | --- |
| `strategy` | `chronological` |
| `timestamp_groups_kept_intact` | `true` |
| `strict_boundaries_verified` | `true` |
| `no_transaction_overlap_verified` | `true` |
| `total_rows` | 10,000 |
| requested fractions | train 0.70, validation 0.15, test 0.15 |
| boundary mode | `fractions_snapped_to_timestamp_groups` |

| Partition | Rows | Timestamp minimum–maximum | Positive labels |
| --- | ---: | --- | ---: |
| Train | 7,011 | `2022-09-01 00:00`–`00:20` | 0 |
| Validation | 1,657 | `2022-09-01 00:21`–`00:25` | 1 |
| Test | 1,332 | `2022-09-01 00:26`–`00:29` | 0 |

The verified `transaction_features.csv` contains all 18 canonical/provenance
columns followed by the 34 engineered fields in Sections 4–7: 52 columns and
10,000 rows in total. The quick split’s sparse class placement makes it a smoke and
leakage proof, not a model-evaluation dataset.

## 9. Full-file audit facts

The following facts came first from an independent read-only scan of every source
row and were reproduced in `artifacts/full/raw_data_audit.json`; they are not
quick-run artifacts.

| Measure | Transactions | Accounts |
| --- | ---: | ---: |
| Data rows | 5,078,345 | 518,581 |
| Valid-width rows | 5,078,345 | 518,581 |
| Missing/empty fields, all columns | 0 | 0 |
| Whitespace-only fields, all columns | 0 | 0 |
| Exact duplicate rows after first | 9 | 0 |
| Unique exact rows | 5,078,336 | 518,581 |

Transaction timestamp minimum/maximum is `2022/09/01 00:00` /
`2022/09/18 16:18`. Labels are `0 = 5,073,168` and `1 = 5,177`. All values in
both amount columns are finite and strictly positive; each has observed range
`0.000001`–`1,046,302,363,293.48`.

Full methodology and file fingerprints are in `data/README.md`.

## 10. Missing, invalid, and duplicate semantics

- Empty text and whitespace-only text are missing for identifier/category fields.
- A non-parsing timestamp is invalid, not a missing date to impute silently.
- A nonnumeric, negative, infinite, or NaN amount is invalid.
- A target outside `{0,1}` is invalid.
- Exact duplicate rows and legitimate repeated interactions are separate concepts.
- Default behavior must preserve source rows and report duplicate evidence; any
  later deduplication requires a versioned decision and before/after counts.
- Validation failure messages must state field and count without dumping sensitive
  row content unnecessarily.

## 11. Fields forbidden as predictors

`is_laundering`, post-event outcomes, split labels, row-derived knowledge of future
events, and any aggregate computed with later timestamps are forbidden predictors.
Entity names and raw row numbers are provenance/context fields, not model signals by
default. Any later exception requires an explicit decision and leakage review.
