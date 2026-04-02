ZIP_NAME="2023CS10848.zip"
TEMP_DIR="temp_sub"
rm -rf $TEMP_DIR
mkdir $TEMP_DIR

cp -r Q1 Q2 Q3 $TEMP_DIR/
cp eval.py icl_starter.py requirements.txt README.md $TEMP_DIR/

find $TEMP_DIR -name "*.jsonl" -type f -delete
find $TEMP_DIR -name "__pycache__" -type d -exec rm -rf {} +

cd $TEMP_DIR
zip -r ../$ZIP_NAME .
cd ..

echo "Submission zip created: $ZIP_NAME"