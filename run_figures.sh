#!/usr/bin/env bash
# ============================================================================
# run_figures.sh
# این اسکریپت خودش محیط پایتون درست را پیدا می‌کند (بین conda و venv های
# مختلف)، بررسی می‌کند کدام‌یک پکیج‌های لازم (torch, torch_geometric,
# osmnx, networkx, sklearn, matplotlib) را نصب دارد، همان را انتخاب و
# اجرا می‌کند، و در نهایت pipeline تولید ۳ نمودار معلق را می‌راند.
#
# اجرا: از ریشهٔ پروژه بزنید:
#   bash run_figures.sh
# ============================================================================
set -e
cd "$(dirname "$0")"

REQUIRED_PKGS=("torch" "torch_geometric" "osmnx" "networkx" "sklearn" "matplotlib")

check_python() {
    local py="$1"
    if [ ! -x "$py" ]; then
        return 1
    fi
    for pkg in "${REQUIRED_PKGS[@]}"; do
        if ! "$py" -c "import $pkg" >/dev/null 2>&1; then
            return 1
        fi
    done
    return 0
}

echo ">> در حال جست‌وجوی محیط‌های پایتون (conda + venv)..."
CANDIDATES=()

# --- ۱) محیط‌های conda ---
if command -v conda >/dev/null 2>&1; then
    while IFS= read -r line; do
        env_path=$(echo "$line" | awk '{print $NF}')
        if [ -x "$env_path/bin/python" ]; then
            CANDIDATES+=("$env_path/bin/python")
        fi
    done < <(conda env list 2>/dev/null | grep -v '^#')
fi

# --- ۲) پوشه‌های venv رایج در کنار پروژه و در خانه ---
for base in . .. ~/Portfolio ~; do
    for name in venv .venv env .env venv310 venv311; do
        cand="$base/$name/bin/python"
        if [ -x "$cand" ]; then
            CANDIDATES+=("$(realpath "$cand")")
        fi
    done
done

# --- ۳) پایتون سیستم و همین پایتون فعال فعلی، به‌عنوان آخرین گزینه ---
CANDIDATES+=("$(command -v python3)")
CANDIDATES+=("$(command -v python)")

echo ">> ${#CANDIDATES[@]} محیط پیدا شد. در حال بررسی هرکدام..."

CHOSEN=""
for py in "${CANDIDATES[@]}"; do
    [ -z "$py" ] && continue
    echo "   بررسی: $py"
    if check_python "$py"; then
        CHOSEN="$py"
        echo "   ✓ همهٔ پکیج‌های لازم اینجا نصب است -- همین انتخاب می‌شود."
        break
    fi
done

if [ -z "$CHOSEN" ]; then
    echo ""
    echo "!! هیچ محیطی با همهٔ پکیج‌های لازم (torch, torch_geometric, osmnx,"
    echo "   networkx, scikit-learn, matplotlib) پیدا نشد."
    echo "   محیط‌هایی که بررسی شدند:"
    for py in "${CANDIDATES[@]}"; do
        [ -n "$py" ] && echo "     - $py"
    done
    echo ""
    echo "   یکی از این محیط‌ها را با نصب پکیج‌های کم کامل کنید، مثلاً:"
    echo "     pip install -r requirements.txt"
    exit 1
fi

echo ""
echo "=================================================================="
echo "در حال اجرای pipeline با: $CHOSEN"
echo "این ممکن است چند دقیقه طول بکشد (دانلود نقشهٔ OSM + آموزش مدل)."
echo "=================================================================="
"$CHOSEN" benchmarks/spanner_pipeline.py

echo ""
echo "=================================================================="
echo "بررسی نمودارهای تولیدشده:"
for fig in global_stretch_cdf.png continual_learning_loss.png directed_asymmetry_evidence.png; do
    if [ -f "results/figures/$fig" ]; then
        echo "  ✓ results/figures/$fig ساخته شد."
    else
        echo "  ✗ results/figures/$fig پیدا نشد -- چیزی در اجرا مشکل داشته."
    fi
done
echo "=================================================================="
echo "اگر هر ۳ مورد بالا ✓ بودند، حالا این را بزنید:"
echo "  git add -A"
echo "  git commit -m \"results: regenerate 3 pending figures with current model\""
echo "  git push origin main"
