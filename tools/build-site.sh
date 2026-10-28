#!/usr/bin/env bash
# Build the published site: the only part of this repository that browsers get
# (GitHub Pages, and the backend at mosquito.charity via the `release` branch).
#
#   SITE_RUN=<deploy run number> SITE_SHA=<commit> [SERVER_URLS=a,b] tools/build-site.sh <out-dir>
#
# The copy is a DENYLIST on purpose: new static folders (js/, vendor/, assets/,
# …) ship automatically, while server code, tests, tooling, CI and docs never
# do. It is then stamped (app-code URLs, version, optionally the backend list),
# minified, and checked for anything that must not be published.
# Needs `npm ci` first (the minifiers are devDependencies).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:?usage: build-site.sh <out-dir>}
SHA=${SITE_SHA:?SITE_SHA (commit SHA) is required}
RUN=${SITE_RUN:?SITE_RUN (deploy run number) is required}

rm -rf "$OUT"
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
# Tracked and new files; never ignored ones (node_modules, caches, local junk).
(cd "$ROOT" && git ls-files -z --cached --others --exclude-standard \
   | while IFS= read -r -d '' f; do [ -f "$f" ] && printf '%s\0' "$f"; done \
   | tar --null -T - -cf -) | tar -xf - -C "$OUT"
cd "$OUT"
# top-level, non-site paths
rm -rf .github .claude tests tools docs site \
       federated_server.py gunicorn.conf.py requirements*.txt Dockerfile Procfile runtime.txt railway.json \
       netlify.toml package.json package-lock.json pyproject.toml pytest.ini \
       .pytest_cache htmlcov .coverage .venv
# anywhere in the tree
find . \( -name '.git*' -o -name node_modules -o -name __pycache__ -o -name '.env*' \) \
     -prune -exec rm -rf {} +
find . -type f -name '*.md' -delete

# Fail loudly rather than publish something broken or private.
test -f index.html || { echo "::error::index.html missing from the site"; exit 1; }
# Regular files and directories only. A symlink would be published as its
# target — by Pages (the artifact dereferences links) and by the backend — so a
# link to /proc/self/environ or DATA_DIR would hand out secrets and user data.
odd=$(find . ! -type f ! -type d -print)
if [ -n "$odd" ]; then echo "::error::symlinks or special files in the site:"; echo "$odd"; exit 1; fi
# Stamp the app-code URLs with this commit so a new index.html can never run
# against a cached copy of the previous js/*.js (version skew).
sed -i -E 's#(src="js/[a-z-]+\.js)\?v=[^"]*"#\1?v='"${SHA::12}"'"#g' index.html
grep -q "js/detector.js?v=${SHA::12}\"" index.html || { echo "::error::app-code version stamp failed"; exit 1; }
# Every deployment shows its own version: X.Y.<run number> plus commit and date.
node "$ROOT/tools/stamp-version.js" index.html "$RUN" "$SHA"
VERSION=$(sed -n -E "s/^const APP_VERSION = '([0-9]+\.[0-9]+\.${RUN})';$/\1/p" index.html)
[ -n "$VERSION" ] || { echo "::error::APP_VERSION stamp failed"; exit 1; }
if [ -n "${SERVER_URLS:-}" ]; then
  node "$ROOT/tools/set-server-urls.js" index.html "$SERVER_URLS"
fi
grep -m1 '^const SERVER_URLS = ' index.html
# Read before minification renames it; the CSP <meta> below allows exactly these.
URLS=$(sed -n -E "s/^const SERVER_URLS = \[(.*)\];$/\1/p" index.html | tr -d "' ")

node "$ROOT/tools/minify-site.js" .
grep -qE "[\"']${VERSION//./\\.}[\"']" index.html || { echo "::error::version $VERSION lost in minification"; exit 1; }
# The page's CSP as a <meta> too, for hosts that can't send headers (Pages).
node "$ROOT/tools/csp-meta.js" index.html "$URLS"
grep -q 'http-equiv="Content-Security-Policy"' index.html || { echo "::error::CSP meta missing"; exit 1; }

for p in federated_server.py gunicorn.conf.py Dockerfile Procfile railway.json requirements.txt package.json tests tools site .github .claude; do
  if [ -e "$p" ]; then echo "::error::$p leaked into the site"; exit 1; fi
done
leaked=$(find . \( -name '*.md' -o -name '.git*' -o -name node_modules -o -name '*.py' -o -name '*.map' \
                 -o -name '*.jsonl' -o -name '.env*' -o -name '.server_secret' -o -name '*.pem' -o -name '*.key' \) -print)
if [ -n "$leaked" ]; then echo "::error::non-site files in the site:"; echo "$leaked"; exit 1; fi

echo "Built v$VERSION: $(find . -type f | wc -l) files, $(du -sh . | cut -f1):"
find . -type f | sort
