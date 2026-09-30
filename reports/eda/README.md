# EDA interpretation guide
Statistical plots use only the 60% training partition. Raw missingness and fixed-rule quality counts use the full input.
Compare default rates with sample counts; small groups are uncertain. Associations do not establish causality.
Review missing employment and interest values, extreme ages, zero income, duplicate rows and label meaning.
Loan grade and interest rate may encode earlier underwriting decisions: confirm they are available at prediction time.
The supplied dataset has no verified borrower ID or observation date: random splitting cannot establish temporal generalization or eliminate unidentified repeat-borrower leakage.
Use grade_intent_counts.csv with the heatmap. The EDA does not independently validate dataset provenance or label definitions.
