# Disclaimer

This is a personal learning project. Please read this before you use, copy or build on it.
It supplements, and does not replace, the [MIT License](LICENSE).

## Educational purpose, no warranty

`treesfs` re-implements decision trees and random forests from scratch to understand how they
work. It is **not** production software and has not been audited, security-reviewed or
validated for any particular purpose. It is provided "as is", without warranty of any kind, as
stated in the [LICENSE](LICENSE). Use it at your own risk. Nothing here is professional,
financial, legal or medical advice.

## Do not use it to make decisions about people

The Adult dataset predicts whether a person's income exceeds $50K from census-derived
attributes that include **sex, race, marital status and native country**, and it comes from the
1994 US census. Models trained on it reproduce the biases of that data and that time. The
models and results in this repository are teaching examples. **They must not be used to make or
support decisions about real people** (for example credit, hiring, housing, insurance or
benefits). Such uses may be regulated or prohibited in your jurisdiction (anti-discrimination,
consumer-credit and data-protection law, and AI-specific regulation), and would need proper
fairness, legal and governance review that this project does not provide.

The impurity-based importances shown in the figures describe what a model used, not what
causes income; they say nothing about individuals.

## Data

* No dataset is stored in this repository. `scripts/download_data.py` fetches the data from the
  original sources, and you are responsible for following their terms (see
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [data/README.md](data/README.md)).
* Both datasets are public and anonymised. Do not try to re-identify anyone in them.
* The licence of this repository (MIT) applies to the code and documentation only, **not** to
  the datasets.

## Results and benchmarks

* Numbers in the README and `reports/` were measured on one machine, on one seed, at one point
  in time, and are meant to be reproduced with the commands in the README, not trusted blindly.
* Timing comparisons with scikit-learn are for learning. scikit-learn is far faster, by design
  (compiled code); this project makes no claim of being competitive with it.
* Accuracy differences between implementations on a single split can come from tie-breaking
  alone (see `docs/DESIGN.md` §9.3).

## Security

The training scripts store models with Python's `pickle`. Never load a pickle file you did not
create yourself. See [SECURITY.md](SECURITY.md).

## No affiliation, trademarks

This project is independent. It is not affiliated with, sponsored by or endorsed by
scikit-learn, NumPy, the UCI Machine Learning Repository, StatLib / Carnegie Mellon University,
or the authors of the datasets and papers it cites. All product and project names are the
property of their owners and are used only to identify them (for example to say what a result
was compared against). "Random Forests" is, to my knowledge, a registered trademark of Leo
Breiman and Adele Cutler; this project uses "random forest" only as the generic name of the
algorithm.

## Not legal advice

This file is a good-faith statement of intent, not legal advice. If you plan to use anything
from this repository commercially or in a regulated setting, get your own advice.
