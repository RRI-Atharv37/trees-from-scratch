# Security policy

`treesfs` is a learning project, but if you find a security problem I would like to hear about
it.

## Reporting a vulnerability

Please **do not open a public issue** for a security problem. Use GitHub's private reporting:
on the repository page open **Security → Report a vulnerability**. I will acknowledge the
report on a best-effort basis (this is a personal project, so there is no service-level
agreement).

## Supported versions

Only the latest release on the default branch is supported.

## Things to know when using this project

* **Pickle files are code.** `scripts/train.py` saves models with Python's `pickle`, and
  `scripts/evaluate.py` loads them. Unpickling a file can execute arbitrary code, so **only load
  `artifacts/*.pkl` files that you created yourself**. Never load a pickle from someone else or
  from the internet. The `artifacts/` folder is gitignored, so none are ever published from
  this repository.
* **Downloads.** `scripts/download_data.py` fetches the Adult zip over HTTPS and refuses it if
  its SHA-256 does not match the value in `configs/default.yaml`. Zip members are read by fixed
  name and never extracted to attacker-chosen paths. California housing is fetched by
  scikit-learn.
* **No secrets.** The project needs no API keys or credentials. If you ever see one committed,
  please report it.
* **Dependencies.** They are pinned in `uv.lock`; update them with `uv lock --upgrade` and
  re-run the tests.
