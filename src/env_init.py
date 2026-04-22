import os
import subprocess
import platform

# These are placeholders that the function will overwrite
GITHUB_TOKEN = "xx"
KAGGLE_USERNAME = "xx"
KAGGLE_KEY = "xx"

def update_env_and_run(github_token=None, kaggle_user=None, kaggle_key=None):
    """
    Updates the tokens in this file and executes the bash setup.
    """
    file_path = "/content/src/env_init.py"
    
    # 1. Update the variables in the file itself if tokens are provided
    if github_token or kaggle_user or kaggle_key:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        with open(file_path, 'w') as f:
            for line in lines:
                if github_token and line.startswith('GITHUB_TOKEN ='):
                    f.write(f'GITHUB_TOKEN = "{github_token}"\n')
                elif kaggle_user and line.startswith('KAGGLE_USERNAME ='):
                    f.write(f'KAGGLE_USERNAME = "{kaggle_user}"\n')
                elif kaggle_key and line.startswith('KAGGLE_KEY ='):
                    f.write(f'KAGGLE_KEY = "{kaggle_key}"\n')
                else:
                    f.write(line)
        print("Tokens updated in env_init.py")

    # 2. Set the Environment Variables for the current session
    os.environ["GITHUB_TOKEN"] = github_token or GITHUB_TOKEN
    os.environ["KAGGLE_USERNAME"] = kaggle_user or KAGGLE_USERNAME
    os.environ["KAGGLE_KEY"] = kaggle_key or KAGGLE_KEY

    # 3. Define and run the bash scripts
    base_drive_path = "/content/drive/MyDrive/apai"
    scripts = [
        f"{base_drive_path}/fetch_src.sh",
        f"{base_drive_path}/setup_colab.sh"
    ]

    for script_path in scripts:
        if os.path.exists(script_path):
            print(f"--- Running {os.path.basename(script_path)} ---")
            
            # Normalize Line Endings (Cross-platform Python fix)
            with open(script_path, 'rb') as f:
                content = f.read().replace(b'\r\n', b'\n')
            with open(script_path, 'wb') as f:
                f.write(content)

            # Execute via Bash
            subprocess.run(["bash", script_path], check=True)
        else:
            print(f"Warning: {script_path} not found.")

if __name__ == "__main__":
    # If run directly without arguments, it uses the existing tokens in the file
    update_env_and_run()