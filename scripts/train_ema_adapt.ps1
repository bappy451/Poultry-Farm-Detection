Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "windows_common.ps1")

function Show-Usage {
    Write-Output "Usage: .\scripts\train_ema_adapt.ps1 [options]"
    Write-Output "Starts from COCO-pretrained YOLO11l-OBB, adapts with EMA on unlabeled NAIP, then fine-tunes on labels."
    Write-Output "Options: --unlabeled-root PATH --data PATH --output-dir PATH --adapt-epochs N --finetune-epochs N --adapt-lr0 N --finetune-lr0 N --freeze-backbone N --batch N --unlabeled-batch N --stage1-only --resume"
    Write-Output 'Example: conda activate ag-foundation; .\scripts\train_ema_adapt.ps1 --unlabeled-root "E:\NAIP_Dataset_PH\MS_tiles_512_256_jpg" --adapt-lr0 1e-5 --finetune-lr0 3e-4 --freeze-backbone 10 --device 0'
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$codeDir = [System.IO.Path]::GetFullPath((Join-Path $scriptDir ".."))
$pythonExe = ""
$showUsage = $false
$forwardArgs = New-Object System.Collections.Generic.List[string]
$originalArgs = @($args)

for ($index = 0; $index -lt $originalArgs.Count; $index++) {
    $token = $originalArgs[$index]
    if ($token -in @("-h", "--help", "-?")) { $showUsage = $true; continue }
    if ($token -eq "--python") {
        if ($index + 1 -ge $originalArgs.Count) { throw "--python requires a path." }
        $pythonExe = $originalArgs[++$index]
        continue
    }
    if ($token -like "--python=*") { $pythonExe = $token.Substring(9); continue }
    $forwardArgs.Add($token) | Out-Null
}

if ($showUsage) { Show-Usage; exit 0 }
$pythonExe = Resolve-PreferredPython -CodeDir $codeDir -RequestedPython $pythonExe
$pythonScript = Join-Path $scriptDir "train_ema_adapt.py"
$exitCode = Invoke-ProjectNativeCommand -Executable $pythonExe -Arguments (@($pythonScript) + $forwardArgs)
exit $exitCode
