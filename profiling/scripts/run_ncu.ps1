$ErrorActionPreference = 'Stop'

$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$BuildDir = Join-Path $RepoRoot "build"
$Config = "Release"
$Target = "flash_attention_00_naive_attention_test"
$TargetExe = Join-Path $BuildDir "csrc\flash-attention\00-naive-attention\$Config\$Target.exe"

if ($env:TINY_CUTLASS_PROFILE_SKIP_BUILD -ne '1') {
    & cmake -S $RepoRoot -B $BuildDir `
        -DTINY_CUTLASS_BUILD_FLASH_ATTENTION=ON `
        -DTINY_CUTLASS_BUILD_CONV_FUSED=OFF
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & cmake --build $BuildDir --config $Config --target $Target
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if ($env:TINY_CUTLASS_PROFILE_SKIP_VERIFY -ne '1') {
    $verifyArgs = @(
        '--batch_size=1'
        '--head_number=1'
        '--head_size=32'
        '--head_size_v=32'
        '--seq_length=32'
        '--seq_length_kv=32'
        '--iterations=1'
        '--reference-check=true'
    )

    & $TargetExe @verifyArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Get-Command ncu -ErrorAction Stop | Out-Null

$iterations = 1
$targetArgs = @(
    '--batch_size=1'
    '--head_number=32'
    '--head_size=256'
    '--head_size_v=256'
    '--seq_length=1024'
    '--seq_length_kv=1024'
    "--iterations=$iterations"
    '--reference-check=false'
)

$reportRoot = Join-Path $BuildDir "reports\profiling\flash-attention\00-naive-attention\ncu"
$csvRoot = Join-Path $reportRoot "csv"
New-Item -ItemType Directory -Force -Path $csvRoot | Out-Null

$reportPath = Join-Path $reportRoot "ncu_00_naive_attention"
$csvPath = Join-Path $csvRoot "ncu_00_naive_attention.csv"

$launchSkipBeforeMatch = 9
$launchCount = 3 * $iterations

$ncuArgs = @(
    '--force-overwrite'
    '--target-processes', 'all'
    '--set', 'detailed'
    '-k', 'regex:attention_.*'
    '--launch-skip-before-match', $launchSkipBeforeMatch
    '--launch-count', $launchCount
    '--page', 'raw'
    '--csv'
    '--import-source=1'
    '--import-sass=1'
    '-o', $reportPath
    '--'
    $TargetExe
) + $targetArgs

Write-Host "[ncu] target exe:"
Write-Host "  $TargetExe"
Write-Host "[ncu] report:"
Write-Host "  $reportPath.ncu-rep"
Write-Host "[ncu] csv:"
Write-Host "  $csvPath"

& ncu @ncuArgs | Tee-Object -FilePath $csvPath
exit $LASTEXITCODE
