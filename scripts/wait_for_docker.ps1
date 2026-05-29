param([int]$maxTries = 60)
$i = 0
for ($i = 0; $i -lt $maxTries; $i++) {
    try {
        docker info | Select-Object -First 20
        Write-Output "Docker is ready"
        exit 0
    } catch {
        Write-Output "waiting Docker... ($i)"
        Start-Sleep -Seconds 2
    }
}
Write-Output "Docker not ready"
exit 1
