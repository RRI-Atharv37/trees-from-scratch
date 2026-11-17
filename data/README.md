# Data

Nothing in `data/raw/` or `data/processed/` is committed (see `.gitignore`); the scripts below recreate it.

```bash
uv run python scripts/download_data.py --dataset all   # or: adult | california
```

## UCI Adult (classification)

- Source: <https://archive.ics.uci.edu/static/public/2/adult.zip> (620,237 bytes).
- SHA-256 (checked on download): `7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb`.
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (as stated on the UCI dataset page). Attribution is given here and in the README. Changes made by this project: parsing, `?` → `missing`, one-hot encoding (see below). Not redistributed.
- Citation: Becker, B. & Kohavi, R. (1996). *Adult*. UCI Machine Learning Repository. <https://doi.org/10.24432/C5XW20>
- Files used: `adult.data` (32,561 rows, 24.08% `>50K`), `adult.test` (16,281 rows, 23.62% `>50K`), `adult.names`.
- Target: `income > 50K` as 0/1.

### Gotchas handled by `data.parse_adult`

- No header row; the 15 column names come from `adult.names`.
- The separator is `", "`: split on `,` and strip every token.
- `adult.test` starts with a junk line `|1x3 Cross validator`: skipped.
- Both files end with a blank line: empty lines are dropped.
- Test labels end in `.` (`<=50K.`, `>50K.`): stripped.
- Missing values are `?`, only in `workclass`, `occupation` and `native-country`.
- `Holand-Netherlands` appears once in train and never in test, so the encoder must cope with rare/unseen categories.
- 24 exact duplicate rows in train are kept, as in the UCI split. A few duplicates carry conflicting labels, so an unlimited tree reaches 0.99997 rather than exactly 1.0 train accuracy.

### Missing-value policy

`?` becomes the explicit category `"missing"` (a one-hot column like any other), so a tree can split on missingness. Rows are never dropped, because dropping them would bias the split. The one-hot encoder is fit on the training split only, caps each column at `data.max_categories` (16) columns with a trailing `other` bucket, and sends unseen test categories to `other` (or an all-zero block if the column has no bucket). Result: 82 features (6 numeric + 76 one-hot).

## California housing (regression)

- Source: StatLib, Pace & Barry (1997), "Sparse spatial autoregressions", *Statistics & Probability Letters* 33(3). Fetched through `sklearn.datasets.fetch_california_housing` (scikit-learn is used as a downloader only; the lazy import lives in `data.download_california`).
- 20,640 rows x 8 features (`MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude`); target = median house value in units of $100k, capped at about 5.0.
- License: **no licence or terms of use are stated** on scikit-learn's page for the loader, and none were found for the original distribution. The data is therefore never committed or redistributed by this repo (`data/raw` and `data/processed` are gitignored); you download it yourself and should cite Pace & Barry. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
- Stored as `data/raw/california_housing.npz`; the seeded 80/20 split (16,512 / 4,128) is cached in `data/processed/california.npz`.
