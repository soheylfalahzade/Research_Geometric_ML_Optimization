#!/usr/bin/env bash
# ============================================================================
# run_figures.sh
# This script finds the right Python environment by itself (among the various conda and venv
# environments), checks which one has the required packages (torch, torch_geometric,
# osmnx, networkx, sklearn, matplotlib) installed, selects it and
# runs it, and finally runs the pipeline that generates the 3 pending figures.
#
# Run from the project root:
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

echo ">> Searching for Python environments (conda + venv)..."
CANDIDATES=()

# --- 1) conda environments ---
if command -v conda >/dev/null 2>&1; then
    while IFS= read -r line; do
        env_path=$(echo "$line" | awk '{print $NF}')
        if [ -x "$env_path/bin/python" ]; then
            CANDIDATES+=("$env_path/bin/python")
        fi
    done < <(conda env list 2>/dev/null | grep -v '^#')
fi

# --- 2) Common venv folders next to the project and in the home directory ---
for base in . .. ~/Portfolio ~; do
    for name in venv .venv env .env venv310 venv311; do
        cand="$base/$name/bin/python"
        if [ -x "$cand" ]; then
            CANDIDATES+=("$(realpath "$cand")")
        fi
    done
done

# --- 3) The system Python and the currently active Python, as the last option ---
CANDIDATES+=("$(command -v python3)")
CANDIDATES+=("$(command -v python)")

echo ">> ${#CANDIDATES[@]} environments found. Checking each one..."

CHOSEN=""
for py in "${CANDIDATES[@]}"; do
    [ -z "$py" ] && continue
    echo "   Checking: $py"
    if check_python "$py"; then
        CHOSEN="$py"
        echo "   ✓ All required packages are installed here -- this one is selected."
        break
    fi
done

if [ -z "$CHOSEN" ]; then
    echo ""
    echo "!! No environment was found with all the required packages (torch, torch_geometric, osmnx,"
    echo "   networkx, scikit-learn, matplotlib)."
    echo "   Environments that were checked:"
    for py in "${CANDIDATES[@]}"; do
        [ -n "$py" ] && echo "     - $py"
    done
    echo ""
    echo "   Complete one of these environments by installing the missing packages, for example:"
    echo "     pip install -r requirements.txt"
    exit 1
fi

echo ""
echo "=================================================================="
echo "Running the pipeline with: $CHOSEN"
echo "This may take a few minutes (OSM map download + model training)."
echo "=================================================================="
"$CHOSEN" benchmarks/spanner_pipeline.py

echo ""
echo "=================================================================="
echo "Checking the generated figures:"
for fig in global_stretch_cdf.png continual_learning_loss.png directed_asymmetry_evidence.png; do
    if [ -f "results/figures/$fig" ]; then
        echo "  ✓ results/figures/$fig was created."
    else
        echo "  ✗ results/figures/$fig was not found -- something went wrong in the run."
    fi
done
echo "=================================================================="
echo "If all 3 items above are ✓, now run this:"
echo "  git add -A"
echo "  git commit -m \"results: regenerate 3 pending figures with current model\""
echo "  git push origin main"
