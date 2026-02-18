#!/usr/bin/env bash
# render-diagrams.sh — Extract mermaid blocks from a markdown file and render each to PNG.
#
# Usage:
#   bash scripts/render-diagrams.sh [DOCS_FILE] [OUTPUT_DIR]
#
# Environment variables (all optional):
#   MMDC                   Path to the mmdc binary          (default: mmdc)
#   PUPPETEER_CONFIG       Path to a puppeteer JSON config  (default: none)
#   PUPPETEER_EXECUTABLE_PATH  Path to Chrome/Chromium      (default: auto-detect)
#   SCALE                  Puppeteer device scale factor    (default: 3)
#   WIDTH                  Viewport width in CSS px         (default: 2400)
#   HEIGHT                 Viewport height in CSS px        (default: 1600)

set -euo pipefail

DOCS_FILE="${1:-docs/use-case-diagrams.md}"
OUTPUT_DIR="${2:-docs/diagrams}"
MMDC="${MMDC:-mmdc}"
SCALE="${SCALE:-3}"
WIDTH="${WIDTH:-2400}"
HEIGHT="${HEIGHT:-1600}"

# ---------------------------------------------------------------------------
# Locate Chrome/Chromium if PUPPETEER_EXECUTABLE_PATH is not already set
# ---------------------------------------------------------------------------
if [ -z "${PUPPETEER_EXECUTABLE_PATH:-}" ]; then
  for candidate in \
      google-chrome-stable \
      google-chrome \
      chromium-browser \
      chromium; do
    if command -v "$candidate" &>/dev/null; then
      export PUPPETEER_EXECUTABLE_PATH
      PUPPETEER_EXECUTABLE_PATH="$(command -v "$candidate")"
      break
    fi
  done
fi

if [ -z "${PUPPETEER_EXECUTABLE_PATH:-}" ]; then
  echo "ERROR: No Chrome/Chromium executable found." >&2
  echo "Set PUPPETEER_EXECUTABLE_PATH or install google-chrome-stable." >&2
  exit 1
fi

echo "Using browser: $PUPPETEER_EXECUTABLE_PATH"

# ---------------------------------------------------------------------------
# Extract mermaid blocks from the markdown file into a temp directory
# ---------------------------------------------------------------------------
TMPDIR_MMD="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_MMD"' EXIT

python3 - "$DOCS_FILE" "$TMPDIR_MMD" <<'PYEOF'
import sys, re, os

md_file, tmp_dir = sys.argv[1], sys.argv[2]

with open(md_file) as f:
    content = f.read()

sections = re.split(r'(?=^## )', content, flags=re.MULTILINE)

for section in sections:
    heading = re.match(r'^## (.+)', section)
    blocks = re.findall(r'```mermaid\n(.*?)```', section, re.DOTALL)
    if not (heading and blocks):
        continue

    raw = heading.group(1).strip()

    # Base name from the script/file name in the heading
    base = re.sub(r'[`]', '', raw)
    suffix_match = re.search(r'[—–]\s*(.+)', raw)
    base = re.sub(r'\s*[—–].*', '', base).strip()
    suffix = (
        '-' + re.sub(r'[^\w]+', '-', suffix_match.group(1)).strip('-').lower()
        if suffix_match else ''
    )
    base = base.replace('.py', '').strip()
    base = re.sub(r'[^\w-]', '-', base).strip('-').lower()
    base = re.sub(r'-+', '-', base)
    fname = base + suffix + '.mmd'

    path = os.path.join(tmp_dir, fname)
    with open(path, 'w') as f:
        f.write(blocks[0])
    print(f"  extracted: {fname}")
PYEOF

# ---------------------------------------------------------------------------
# Build mmdc argument list
# ---------------------------------------------------------------------------
MMDC_ARGS=("-w" "$WIDTH" "-H" "$HEIGHT" "-s" "$SCALE" "--backgroundColor" "white")
if [ -n "${PUPPETEER_CONFIG:-}" ]; then
  MMDC_ARGS+=("-p" "$PUPPETEER_CONFIG")
fi

mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Render each diagram
# ---------------------------------------------------------------------------
shopt -s nullglob
MMD_FILES=("$TMPDIR_MMD"/*.mmd)
if [ "${#MMD_FILES[@]}" -eq 0 ]; then
  echo "ERROR: No mermaid blocks found in $DOCS_FILE" >&2
  exit 1
fi

for mmd_file in "${MMD_FILES[@]}"; do
  name="$(basename "$mmd_file" .mmd)"
  out="$OUTPUT_DIR/$name.png"
  echo "  rendering $name → $out"
  "$MMDC" -i "$mmd_file" -o "$out" "${MMDC_ARGS[@]}"
done

echo "Done — ${#MMD_FILES[@]} diagram(s) written to $OUTPUT_DIR/"
