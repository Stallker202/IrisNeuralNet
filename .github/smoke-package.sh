#!/usr/bin/env bash
# Проверка собранного архива: файлы на месте, консольный режим считает,
# веб-лаборатория стартует не из своей папки, отдаёт страницу и обучает сеть по HTTP.
# Использование: smoke-package.sh <папка публикации>
set -euo pipefail

dir="$(cd "$1" && pwd)"
exe="$dir/IrisNeuralNet.Lab"
for file in "$exe" "$dir/wwwroot/index.html" "$dir/wwwroot/lab.js" "$dir/iris.csv" "$dir/sonar.csv" "$dir/boston.csv"; do
  [ -e "$file" ] || { echo "нет файла: $file"; exit 1; }
done

work="$(mktemp -d)"
cd "$work"
"$exe" --console iris --quick --no-charts --out "$work/reports" > console.log
grep -q "Отчёт:" console.log

port=5299
url="http://127.0.0.1:$port"
"$exe" --no-browser --port "$port" > server.log 2>&1 &
server=$!
trap 'kill "$server" 2>/dev/null || true' EXIT

for _ in $(seq 60); do curl -fsS "$url/api/catalog" -o catalog.json 2>/dev/null && break; sleep 1; done
grep -q '"iris"' catalog.json
curl -fsS "$url/" | grep -q "<title>"
curl -fsS "$url/theme.css" | grep -q -- "--background"

json='Content-Type: application/json'
curl -fsS -X POST -H "$json" "$url/api/data" \
  -d '{"dataset":"iris","scaling":"Standardize","validationFraction":0.2,"learningRate":0.01}'
curl -fsS -X POST -H "$json" "$url/api/train/start" \
  -d '{"hidden":"8","optimizer":"Adam","learningRate":0.01,"epochs":20,"batchSize":10,"earlyStoppingPatience":0,"keepPrevious":false}'
for _ in $(seq 60); do
  curl -fsS "$url/api/state" | grep -q '"running":false' && break
  sleep 1
done
curl -fsS "$url/api/state" | grep -q 'epoch 20'
curl -fsS "$url/api/view/loss.svg?w=640&h=360" | grep -q "<polyline"
echo "smoke OK: $dir"
