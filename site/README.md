# site/ — the built app (generated, do not edit)

Empty on `main`. The Pages deploy (`.github/workflows/deploy-pages.yml`) builds
the minified PWA with `tools/build-site.sh` and commits it here on the
`release` branch, which is the branch Railway deploys. `federated_server.py`
then also serves the app at `/` (https://mosquito.charity). With no
`site/index.html` the server is API-only.
