# PowerShell script to set up a virtual environment with Python 3.11.9

# Check if Python 3.11.9 is installed
$pythonPath = "python"
$pythonVersion = & $pythonPath --version 2>&1

if ($pythonVersion -notmatch "3\.11\.9") {
    Write-Host "Python 3.11.9 is not detected. Please install Python 3.11.9 before running this script." -ForegroundColor Red
    Write-Host "You can download it from: https://www.python.org/downloads/release/python-3119/" -ForegroundColor Yellow
    exit 1
}

Write-Host "Python 3.11.9 detected. Setting up virtual environment..." -ForegroundColor Green

# Create a virtual environment
$venvName = "venv-py3119"
Write-Host "Creating virtual environment: $venvName" -ForegroundColor Cyan
& $pythonPath -m venv $venvName

# Activate the virtual environment and install dependencies
Write-Host "Activating virtual environment and installing dependencies..." -ForegroundColor Cyan
$activateScript = ".\$venvName\Scripts\Activate.ps1"

# Check if activation script exists
if (Test-Path $activateScript) {
    # Create a temporary script to activate and install
    $tempScript = "temp_install.ps1"
    
    @"
    & '$activateScript'
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    Write-Host "Dependencies installed successfully!" -ForegroundColor Green
    Write-Host "\nTo activate this environment in the future, run:" -ForegroundColor Yellow
    Write-Host ".\$venvName\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host "\nTo deactivate the environment, simply run:" -ForegroundColor Yellow
    Write-Host "deactivate" -ForegroundColor Yellow
"@ | Out-File -FilePath $tempScript -Encoding utf8
    
    # Execute the temporary script in a new PowerShell process
    & powershell -ExecutionPolicy Bypass -File $tempScript
    
    # Clean up
    Remove-Item $tempScript
} else {
    Write-Host "Error: Virtual environment activation script not found at $activateScript" -ForegroundColor Red
    exit 1
}