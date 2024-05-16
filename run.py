import subprocess
import time

def run_script(script_name):
    try:
        result = subprocess.run(['python', script_name], check=True, text=True, capture_output=True)
        print(f"Output of {script_name}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script_name}:\n{e.stderr}")

if __name__ == "__main__":
    # Run detection.py
    run_script('detection.py')
    
    # Run pca.py
    run_script('pca.py')
    
    # Run direction.py
    run_script('direction.py')
