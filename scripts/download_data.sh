#!/bin/bash
# Download the public RecGen training renderings and lay them out for training.
#
# Usage: ./scripts/download_data.sh <output_dir> [ss|slat|all] [ABO|HSSD|Objaverse|PartNeXt|all]
#
# The data has two collections: ss/ (stage-1, sparse structure) and slat/
# (stage-2, structured latent). This downloads the gzipped WebDataset shards,
# decompresses them, and renumbers each collection/subset contiguously (the
# published shard numbering has gaps, which brace-expansion patterns cannot
# express). The original shard index of every renumbered file is recorded in
# <collection>/<subset>/shard_map.json.
set -euo pipefail

OUT="${1:?usage: ./scripts/download_data.sh <output_dir> [ss|slat|all] [subset|all]}"
STAGE="${2:-all}"
SUBSET="${3:-all}"
BASE="https://tri-ml-public.s3.amazonaws.com/github/recgen/train"
PAR="${PAR:-8}"

command -v pigz >/dev/null && GUNZIP="pigz -d" || GUNZIP="gunzip"

mkdir -p "$OUT"
curl -sfL "$BASE/INDEX.txt" -o "$OUT/INDEX.txt"

want() { # key -> 0/1
  local k="$1" stage="${1%%/*}" rest sub
  case "$k" in LICENSES/*|README.md) return 0;; esac
  [ "$STAGE" != all ] && [ "$stage" != "$STAGE" ] && return 1
  case "$stage" in ss|slat) ;; *) return 1;; esac
  if [ "$SUBSET" != all ]; then
    rest="${k#*/}"; sub="${rest%%/*}"
    [ "$sub" = "$SUBSET" ] || { [ "$rest" = blacklist.txt ] || return 1; }
  fi
  return 0
}

# 1) download + decompress (resumable: skips existing outputs; collections
#    already renumbered — shard_map.json present — are skipped)
grep -v '^$' "$OUT/INDEX.txt" | while read -r key; do
  want "$key" || continue
  dir="$(dirname "$key")"
  [ -e "$OUT/$dir/shard_map.json" ] && case "$key" in */shard-*.tar.gz) continue;; esac
  echo "$key"
done | xargs -P "$PAR" -I{} sh -c '
  key="{}"; dst="'"$OUT"'/$key"
  plain="${dst%.gz}"
  [ -e "$plain" ] && exit 0
  mkdir -p "$(dirname "$dst")"
  curl -sfL "'"$BASE"'/$key" -o "$dst.part" || { echo "FAILED $key" >&2; exit 1; }
  mv "$dst.part" "$dst"
  case "$dst" in *.tar.gz) '"$GUNZIP"' "$dst";; esac
'

# 2) renumber shards contiguously per collection/subset and write shard_map.json
for stage in ss slat; do
  [ "$STAGE" != all ] && [ "$STAGE" != "$stage" ] && continue
  for d in ABO HSSD Objaverse PartNeXt; do
    [ "$SUBSET" != all ] && [ "$SUBSET" != "$d" ] && continue
    [ -d "$OUT/$stage/$d" ] || continue
    python3 - "$OUT/$stage/$d" <<'EOF'
import json, os, re, sys
root = sys.argv[1]
shards = sorted(f for f in os.listdir(root) if re.fullmatch(r'shard-\d{6}\.tar', f))
mapping, tmp = {}, []
for new, name in enumerate(shards):
    dst = f'shard-{new:06d}.tar'
    mapping[dst] = name
    if name != dst:
        os.rename(os.path.join(root, name), os.path.join(root, name + '.renum'))
        tmp.append((name + '.renum', dst))
for src, dst in tmp:
    os.rename(os.path.join(root, src), os.path.join(root, dst))
json.dump(mapping, open(os.path.join(root, 'shard_map.json'), 'w'), indent=0)
print(f'{root}: {len(shards)} shards -> shard-000000..shard-{len(shards)-1:06d}')
EOF
  done
done

echo "Done. Data in $OUT"
