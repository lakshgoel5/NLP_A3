lang=$1
test_file=$2
output_dir=${3:-./output}
python infer.py --lang "$lang" --test_file "$test_file" --output_dir "$output_dir" --config_path ./config.json
