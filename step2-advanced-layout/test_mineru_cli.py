import os
import subprocess
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run MinerU CLI directly for testing.")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("-o", "--output_dir", default="mineru_outputs", help="Output directory")
    args = parser.parse_args()

    pdf_path = os.path.abspath(args.pdf_path)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' does not exist.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Running MinerU extraction on: {pdf_path}")
    print(f"Output directory: {output_dir}")
    
    # Construct the command
    command = [
        "mineru",
        "-p", pdf_path,
        "-o", output_dir,
        "-b", "pipeline"
    ]

    print(f"Command: {' '.join(command)}")

    try:
        # Run the command and print output directly
        # If running on Windows, shell=True might be needed if mineru is a .bat/.cmd script, 
        # but let's try without it first or use shutil.which
        import shutil
        mineru_cmd = shutil.which("mineru") or "mineru"
        command[0] = mineru_cmd

        subprocess.run(command, check=True)
        print(f"\nExtraction completed successfully! Outputs are in {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"\nError: MinerU command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("\nError: 'mineru' command not found. Please ensure it is installed and in your PATH.")
        print("You can install it using: uv pip install -U \"mineru[all]\"")
        sys.exit(1)

if __name__ == "__main__":
    main()
