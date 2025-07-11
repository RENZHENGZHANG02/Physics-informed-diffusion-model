import argparse
import yaml
import json
import os
import pandas as pd 

def main(task_dict_json, task_types_json, output_path="configs/task_registry.yaml"):
    task_dict = json.loads(task_dict_json)
    task_types = json.loads(task_types_json)
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(this_dir, ".."))
    output_path = os.path.join(project_root, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    registry = {}
    for task, cols in task_dict.items():
        registry[task] = {
            "types": task_types[task],
            "cols": cols,
            "num_classes": {}
        }

        df = pd.read_csv(f"../data/raw/{task}.csv.gz")
        for i, col_type in enumerate(task_types[task]):
            col_name = cols[i]
            if col_type == "classification":
                registry[task]["num_classes"][col_name] = df[col_name].nunique()
            else:
                registry[task]["num_classes"][col_name] = 0

    with open(output_path, "w") as f:
        yaml.dump(registry, f, sort_keys=False)
    print(f"Generated {output_path} successfully.")

def load_registry(path):
    with open(path, "r") as f:
        registry = yaml.safe_load(f)
    return registry

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate task registry YAML.")
    parser.add_argument("--task_dict", type=str, required=True,
                        help='JSON string of task to cols, e.g. \'{"cv":["Cv"]}\'')
    parser.add_argument("--task_types", type=str, required=True,
                        help='JSON string of task to type, e.g. \'{"cv":"regression"}\'')
    parser.add_argument("--output", type=str, default="configs/task_registry.yaml",
                        help="Output path for registry.yaml")

    args = parser.parse_args()
    main(args.task_dict, args.task_types, args.output)
