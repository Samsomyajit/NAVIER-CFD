# NAVIER-CFD website deployment

The project site is built from `website/` and `docs/` and published to the generated `gh-pages` branch by `.github/workflows/pages-branch.yml`.

GitHub Pages should be configured once under **Settings → Pages** with:

- **Source:** Deploy from a branch
- **Branch:** `gh-pages`
- **Folder:** `/ (root)`

The former `actions/configure-pages` deployment was removed because this repository did not yet have a Pages site object and GitHub returned `Get Pages site failed: Not Found`.

Published routes:

- `/NAVIER-CFD/`
- `/NAVIER-CFD/recommender/`
- `/NAVIER-CFD/docs/`
