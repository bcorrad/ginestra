import os
import pandas as pd
import re
import ast

def parse_metrics_line(line):
    try:
        config_match = re.search(r'\[CONFIG (\d+)', line)
        config = int(config_match.group(1)) if config_match else None

        values = re.findall(r'([\d.]+) ± ([\d.]+)', line)
        if len(values) != 7:
            return None

        flat_values = [float(val) for pair in values for val in pair]
        return [config] + flat_values
    except Exception as e:
        print(f"Failed to parse metrics line: {line}\nError: {e}")
        return None

def extract_config_dict(lines):
    for line in lines:
        if "Model configuration:" in line:
            try:
                config_str = line.split("Model configuration:", 1)[-1].strip()
                config_dict = ast.literal_eval(config_str)
                return config_dict
            except Exception as e:
                print(f"Failed to parse config dict: {line}\nError: {e}")
                return None
    return None

def extract_num_params(lines):
    for line in lines:
        if "Number of parameters:" in line:
            try:
                match = re.search(r'Number of parameters:\s*(\d+)', line)
                if match:
                    return int(match.group(1))
            except Exception as e:
                print(f"Failed to parse parameter count: {line}\nError: {e}")
    return None

def process_experiment(folder_path):
    data = []
    experiment_name = os.path.basename(folder_path)
    model_name = experiment_name.split('_')[0].upper()
    reports_path = os.path.join(folder_path, 'reports')

    if not os.path.isdir(reports_path):
        print(f"Skipping {folder_path}: no 'reports' folder found.")
        return

    files = [f for f in os.listdir(reports_path) if os.path.isfile(os.path.join(reports_path, f))]

    for file_name in files:
        file_path = os.path.join(reports_path, file_name)
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if not lines:
                    continue

                metrics_line = next((l for l in reversed(lines) if l.strip()), "")
                parsed_metrics = parse_metrics_line(metrics_line)
                config_dict = extract_config_dict(lines)
                num_params = extract_num_params(lines)

                if parsed_metrics:
                    config = parsed_metrics[0]
                    metric_values = parsed_metrics[1:]
                    config_str = str(config_dict) if config_dict else ""

                    row = [
                        experiment_name,
                        model_name,
                        "",  # N.Node Features
                        "",  # Notes
                        config,
                        config_str,
                        num_params
                    ] + metric_values

                    data.append(row)
        except Exception as e:
            print(f"Error reading {file_name} in {experiment_name}: {e}")

    if data:
        columns = [
            "Experiment ID", "Model", "N.Node Features", "Notes",
            "Config", "Config Dict", "Num Params",
            "Val Loss Avg", "Val Loss Std",
            "Val Precision Avg", "Val Precision Std",
            "Val Recall Avg", "Val Recall Std",
            "Val F1 Avg", "Val F1 Std",
            "Val Top-1 Avg", "Val Top-1 Std",
            "Val Top-3 Avg", "Val Top-3 Std",
            "Val Top-5 Avg", "Val Top-5 Std"
        ]
        df = pd.DataFrame(data, columns=columns)
        df.sort_values(by="Val Precision Avg", ascending=False, inplace=True)

        output_csv = os.path.join(folder_path, f"{experiment_name}.csv")
        df.to_csv(output_csv, index=False)
        print(f"Saved CSV for {experiment_name} at {output_csv}")

def process_all_experiments_parent(root_path):
    for entry in os.listdir(root_path):
        experiment_path = os.path.join(root_path, entry)
        if os.path.isdir(experiment_path):
            process_experiment(experiment_path)
            
def process_all_experiments(root_path):
    if os.path.isdir(root_path):
        process_experiment(root_path)

if __name__ == "__main__":
    # Run it
    root = '/repo/corradini/ginestra/experiments/'
    process_all_experiments_parent(root)
