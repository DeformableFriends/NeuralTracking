
SPLIT="test"

echo
echo "GENERATE..."
python generate.py --split=${SPLIT}

echo
echo "EVALUATE..."
python evaluate.py --split=${SPLIT}