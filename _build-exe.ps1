# Install Poetry if not already installed
if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    pip install poetry
}

# Install project dependencies with cli extras
poetry install --extras "cli"

# Install pyinstaller in the poetry environment
poetry run pip install pyinstaller

# Create PyInstaller spec file with specified data collections
# poetry run pyi-makespec --collect-data=gradio_client --collect-data=gradio rembg.py

# Run PyInstaller with the generated spec file
poetry run pyinstaller rembg.spec
