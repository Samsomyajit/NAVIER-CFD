# Deployment

The repository contains GitHub Actions workflows for Python 3.10–3.12 continuous integration, MkDocs Material documentation deployment to GitHub Pages, and tagged package builds.

- Documentation target: https://samsomyajit.github.io/NAVIER-CFD/
- CI workflow: `.github/workflows/ci.yml`
- Documentation workflow: `.github/workflows/docs.yml`
- Distribution workflow: `.github/workflows/package.yml`

If the GitHub App commit does not automatically start Actions, open the **Actions** tab and run **Deploy documentation** once. Subsequent normal pushes to `main` deploy automatically.
