import json
import sys

def validate_schema(file_path):
    print(f"\n--- Validating file: {file_path} ---")
    expected_top_keys = ["articleId", "sentId", "sentText", "relationMentions"]
    expected_rel_keys = ["em1Text", "em2Text", "label"]
    
    lines_checked = 0
    errors = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            print(f"Line {line_num}:")
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print("  [FAIL] Invalid JSON format.")
                errors += 1
                continue
                
            # Check top-level keys
            for key in expected_top_keys:
                if key in obj:
                    print(f"  [PASS] Top-level key '{key}' is present.")
                else:
                    print(f"  [FAIL] Top-level key '{key}' is MISSING.")
                    errors += 1
            
            # Check relationMentions
            if "relationMentions" in obj:
                if not isinstance(obj["relationMentions"], list):
                    print("  [FAIL] 'relationMentions' is not a list.")
                    errors += 1
                else:
                    for idx, rm in enumerate(obj["relationMentions"]):
                        print(f"  Relation Mention [{idx}]:")
                        for rel_key in expected_rel_keys:
                            if rel_key in rm:
                                print(f"    [PASS] Key '{rel_key}' is present.")
                            else:
                                print(f"    [FAIL] Key '{rel_key}' is MISSING.")
                                errors += 1
            
            lines_checked += 1

    if errors == 0:
        print(f"-> SUCCESS! {lines_checked} lines passed all checks perfectly.\n")
    else:
        print(f"-> FAILED with {errors} errors across {lines_checked} lines.\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_output.py <file.jsonl>")
        sys.exit(1)
        
    for f in sys.argv[1:]:
        validate_schema(f)