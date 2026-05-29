$me = $PID
$ps = Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $me -and $_.CommandLine -and $_.CommandLine -match 'src.train' }
if ($ps) {
  $ps | ForEach-Object {
    Write-Output "Killing PID:$($_.ProcessId) - $($_.CommandLine)"
    try {
      Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop
      Write-Output "Killed PID:$($_.ProcessId)"
    } catch {
      Write-Output "Failed to kill PID:$($_.ProcessId): $_"
    }
  }
} else {
  Write-Output 'No src.train processes found'
}
