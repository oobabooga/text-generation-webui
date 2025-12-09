# Clone or update all repositories for MerwelLabs into the specified destination
$dest = 'D:\OneDrive_Merwel\OneDrive\Github'
$gh = 'C:\Program Files\GitHub CLI\gh.exe'

if (!(Test-Path $gh)) {
    Write-Error "gh not found at $gh. Ensure GitHub CLI is installed and on PATH."
    exit 1
}

New-Item -ItemType Directory -Path $dest -Force | Out-Null

Write-Output "Listing repositories for MerwelLabs..."
$reposJson = & $gh repo list MerwelLabs --limit 1000 --json name,sshUrl,visibility
if (-not $reposJson) {
    Write-Output "No repos found or gh returned nothing."
    exit 0
}

$repos = $reposJson | ConvertFrom-Json

foreach ($r in $repos) {
    $dir = Join-Path $dest $r.name
    if (Test-Path $dir) {
        Write-Output "Updating $($r.name)..."
        try {
            git -C $dir pull --rebase
        } catch {
            Write-Warning "Failed to pull $($r.name): $_"
        }
    } else {
        Write-Output "Cloning $($r.name)..."
        try {
            git clone $r.sshUrl $dir
        } catch {
            Write-Warning "Failed to clone $($r.name): $_"
        }
    }
}

Write-Output "Done. Repositories are in: $dest"
