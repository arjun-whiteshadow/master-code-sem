# Growth log: column definitions

The growth log is a spreadsheet with one row per sample, placed at
`method1/data/growth_conditions.xlsx` (the path is set in `config.yaml`).
`make_log_template.py` writes an empty copy with the sample IDs of the
current feature table filled in. The file is not committed.

| Column | Unit | Type | Meaning |
|---|---|---|---|
| `Sample` | – | text | Sample ID, spelled exactly as in the feature table (for example `111623DN06`) |
| `Substrate_Temperature_C` | °C | numeric | Substrate temperature during nanowire growth, as read from the thermocouple or pyrometer used in the growth record |
| `Ga_BEP_Torr` | Torr | numeric | Gallium beam equivalent pressure |
| `As_BEP_Torr` | Torr | numeric | Arsenic beam equivalent pressure |
| `Sb_BEP_Torr` | Torr | numeric | Antimony beam equivalent pressure |
| `Growth_Time_min` | min | numeric | Duration of nanowire growth, excluding oxide desorption and any pre-deposition step |
| `Substrate` | – | category | Substrate type, one label per kind (for example `Si(111)`, `graphene/Si`) |
| `Notes` | – | text | Anything unusual about the run; not analysed |

Rules for filling it in:

- Leave a cell blank when the value is not in the growth record. Do not
  estimate it. A parameter with fewer than 80% of samples filled is left out
  of the analysis and the exclusion is reported.
- Numeric columns hold numbers only; a unit written into a cell (`550 C`)
  stops the run so that it can be corrected.
- Use one spelling per substrate label. `Si(111)` and `Si (111)` count as two
  substrates.
- If the same quantity was recorded in different ways for different runs
  (a pyrometer for some, a thermocouple for others), say so in `Notes` and
  raise it before the analysis is run.
- A parameter that has the same value in every run cannot explain anything
  and is dropped automatically. Add a column only for something that varied.

Columns can be renamed or added; the names in `config.yaml` under
`parameters` must match the sheet.
