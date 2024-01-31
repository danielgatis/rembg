# Install required packages
pip install pyinstaller
pip install -e ".[cli]"

# Create PyInstaller spec file with specified data collections
# pyi-makespec --collect-data=gradio_client --collect-data=gradio rembg.py

# Run PyInstaller with the generated spec file
pyinstaller rembg.spec
