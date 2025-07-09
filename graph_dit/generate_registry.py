import argparse
import yaml
import json
import os

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
            "type": task_types[task], 
            "cols": cols
        }

    with open(output_path, "w") as f:
        yaml.dump(registry, f, sort_keys=False)
    print(f"Generated {output_path} successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate task registry YAML.")
    parser.add_argument("--task_dict", type=str, required=True,
                        help='JSON string of task to cols, e.g. \'{"cv":["Cv"], "upve":["zpve"]}\'')
    parser.add_argument("--task_types", type=str, required=True,
                        help='JSON string of task to type, e.g. \'{"cv":"regression", "upve":"regression"}\'')
    parser.add_argument("--output", type=str, default="configs/task_registry.yaml",
                        help="Output path for registry.yaml")

    args = parser.parse_args()
    main(args.task_dict, args.task_types, args.output)
