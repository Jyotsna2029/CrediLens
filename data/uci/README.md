# Data provenance
Source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
Download: https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip
Citation: Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H.
License: CC BY 4.0 (attribution required).

The CSV is a conversion of the original Excel sheet using pandas.read_excel(header=1). No values were changed during conversion. Original ZIP retained.
ID is excluded from predictors. The training adapter renames `default payment next month` to `loan_status` internally. These are different outcomes across the two datasets, despite the shared internal column name.
Amounts are in New Taiwan dollars. Payment history covers April–September 2005. Preserve category codes, including values not documented on the overview page; do not invent an interpretation for them.
Exact duplicate feature+target rows (excluding ID) are removed before splitting. Age, sex, marriage and education require careful interpretation; no fairness guarantee is claimed. This is a historical benchmark, not validation for Indian lending.

Identical predictor profiles, including conflicting-label cases, are grouped into the same train/validation/test partition. Optional cross-validation also respects these groups.
