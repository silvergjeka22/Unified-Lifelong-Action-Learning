import os
import subprocess
import platform
from google.colab import drive

def initialize_environment():
    # 1. Connect with Drive
    if not os.path.exists('/content/drive'):
        drive.mount('/content/drive')

    # 2. Set Environment Variables
    os.environ["GITHUB_TOKEN"] = "xx"
    os.environ["KAGGLE_USERNAME"] = "APAI_colab_token"
    os.environ["KAGGLE_KEY"] = "xx"

    # 3. Identify OS
    current_os = platform.system()
    print(f"Detected OS: {current_os}")

    # 4. Define Scripts
    scripts = [
        "/content/drive/MyDrive/apai/fetch_src.sh",
        "/content/drive/MyDrive/apai/setup_colab.sh"
    ]

    for script_path in scripts:
        if os.path.exists(script_path):
            print(f"--- Preparing {os.path.basename(script_path)} ---")
            
            # Clean CRLF only on Linux/macOS
            if current_os in ["Linux", "Darwin"]:  # Darwin is macOS
                # Use a cross-platform Python way to strip \r instead of 'sed' 
                # to avoid the sed -i vs sed -i '' headache
                with open(script_path, 'rb') as f:
                    content = f.read()
                
                clean_content = content.replace(b'\r\n', b'\n')
                
                with open(script_path, 'wb') as f:
                    f.write(clean_content)
                
                print(f"  Line endings normalized to LF.")

            # Execute the bash script
            # In Colab/Linux/Mac we use bash. On Windows, we'd need a shell like Git Bash.
            try:
                subprocess.run(["bash", script_path], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error executing script: {e}")
        else:
            print(f"Warning: Script not found at {script_path}")

    # 5. Handle imports.py
    imports_path = "/content/src/imports.py"
    if os.path.exists(imports_path):
        print(f"--- Environment ready. Execute %run {imports_path} in your notebook cell ---")
    else:
        print("Note: /content/src/imports.py not found. Check your bash script logs.")

if __name__ == "__main__":
    initialize_environment()