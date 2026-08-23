import copy
import sys
import subprocess
import yaml
import tempfile
from pathlib import Path

def run_label_efficiency_sweep():
    """
    Automates the nested label-efficiency curve experiment for the DINO-to-YOLO distilled-backbone pipeline.
    Runs the downstream comparison script for each fraction.
    """
    base_config_path = Path("configs/downstream_scratch_compare.yaml")
    
    if not base_config_path.exists():
        print(f"Error: Base config {base_config_path} not found.")
        return
        
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
        
    datasets = ['poultry', 'grsd']
    # Full sweep: 1%, 5%, 10%, 25%, 50%, 100%
    fractions = ['1percent', '5percent', '10percent', '25percent', '50percent', '100percent']
    
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"Starting Label-Efficiency Sweep for: {dataset.upper()}")
        print(f"{'='*80}")
        
        for frac in fractions:
            if frac == '100percent':
                data_yaml = f"configs/{dataset}_80_10_10.yaml"
            else:
                data_yaml = f"configs/{dataset}_nested_{frac}.yaml"
                
            if not Path(data_yaml).exists():
                print(f"  Warning: YAML {data_yaml} not found. Did you run create_nested_subsets.py?")
                continue
                
            print(f"\n>>> Running {frac} subset using {data_yaml} <<<")
            
            run_config = copy.deepcopy(base_config)
            
            # The DINO-distilled comparison script takes the dataset via data_yaml path
            if 'paths' not in run_config:
                run_config['paths'] = {}
                
            run_config['paths']['data_yaml'] = data_yaml
            run_config['paths']['output_dir'] = f"runs/label_efficiency_{dataset}/{frac}"
            
            with tempfile.NamedTemporaryFile('w', delete=False, suffix='.yaml') as tmp_file:
                yaml.dump(run_config, tmp_file, sort_keys=False)
                tmp_config_path = tmp_file.name
                
            try:
                # Run the DINO-distilled downstream comparison script
                import os
                env = os.environ.copy()
                env["PYTHONPATH"] = str(Path("src").absolute())
                python_exe = sys.executable
                cmd = [python_exe, "scripts/downstream_distill_compare.py", "--config", tmp_config_path]
                subprocess.run(cmd, check=True, env=env)
            except subprocess.CalledProcessError as e:
                print(f"Error executing {frac} run for {dataset}: {e}")
            finally:
                Path(tmp_config_path).unlink()

if __name__ == "__main__":
    run_label_efficiency_sweep()
