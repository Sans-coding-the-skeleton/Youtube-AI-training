$jsonFiles = Get-ChildItem "dataset\*.info.json"
$webpFiles = Get-ChildItem "dataset\*.webp"

# Create a hash set for faster lookup
$webpNames = @{}
foreach ($w in $webpFiles) {
    $webpNames[$w.BaseName] = $true
}

$validIds = @()
foreach ($j in $jsonFiles) {
    $id = $j.Name -replace '\.info\.json$', ''
    if ($webpNames.ContainsKey($id)) {
        $validIds += $id
    }
}

Write-Host "Valid pairs found: $($validIds.Count)"

$target = 1500
$keepIds = @{}
for ($i = 0; $i -lt [math]::Min($target, $validIds.Count); $i++) {
    $keepIds[$validIds[$i]] = $true
}

Write-Host "Retaining $target valid pairs..."

# Remove any file that isn't part of the exactly 1500 kept pairs
$allFiles = Get-ChildItem "dataset\*.*"
$removedCount = 0
foreach ($file in $allFiles) {
    $id = $file.Name -replace '\.info\.json$|\.webp$|\.jpg$', ''
    if (-not $keepIds.ContainsKey($id)) {
        Remove-Item $file.FullName -Force
        $removedCount++
    }
}

Write-Host "Removed $removedCount orphan/excess files."

$jCount = (Get-ChildItem "dataset\*.info.json").Count
$wCount = (Get-ChildItem "dataset\*.webp").Count
$jCountOld = (Get-ChildItem "dataset\*.jpg").Count

Write-Host "Final JSON count: $jCount"
Write-Host "Final WEBP count: $wCount"
Write-Host "Final JPG count: $jCountOld"
