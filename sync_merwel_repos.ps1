<#
PowerShell script to sync MerwelLabs GitHub repositories into the OneDrive folder.
Writes logs to D:\OneDrive_Merwel\OneDrive\Github\sync_logs and keeps the last 30 logs.
#>

$dest = 'D:\OneDrive_Merwel\OneDrive\Github'
$gh = 'C:\Program Files\GitHub CLI\gh.exe'
$logDir = Join-Path $dest 'sync_logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$timestamp = (Get-Date).ToString('yyyy-MM-dd_HHmmss')
$logFile = Join-Path $logDir "sync_$timestamp.log"

Start-Transcript -Path $logFile -Force

Write-Output "=== MerwelLabs repo sync started: $(Get-Date -Format o) ==="

if (!(Test-Path $gh)) {
    Write-Error "gh not found at $gh. Exiting."
    Stop-Transcript
    exit 1
}

Write-Output "Listing repositories for MerwelLabs..."
$reposJson = & $gh repo list MerwelLabs --limit 1000 --json name,sshUrl,visibility
if (-not $reposJson) {
    Write-Output "No repos found or gh returned nothing."
    Stop-Transcript
    exit 0
}

$repos = $reposJson | ConvertFrom-Json

foreach ($r in $repos) {
    $dir = Join-Path $dest $r.name
    if (Test-Path $dir) {
        Write-Output "Updating $($r.name)..."
        try {
            git -C $dir pull --rebase 2>&1 | ForEach-Object { Write-Output $_ }
        } catch {
            Write-Warning "Failed to pull $($r.name): $_"
        }
    } else {
        Write-Output "Cloning $($r.name)..."
        try {
            git clone $r.sshUrl $dir 2>&1 | ForEach-Object { Write-Output $_ }
        } catch {
            Write-Warning "Failed to clone $($r.name): $_"
        }
    }
}

Write-Output "=== MerwelLabs repo sync finished: $(Get-Date -Format o) ==="
Stop-Transcript

# Rotate logs: keep last 30 log files
Get-ChildItem -Path $logDir -Filter 'sync_*.log' | Sort-Object LastWriteTime -Descending | Select-Object -Skip 30 | Remove-Item -Force -ErrorAction SilentlyContinue
