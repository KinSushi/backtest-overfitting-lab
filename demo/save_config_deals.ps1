#Requires -Version 5.1
<#
.SYNOPSIS
    Archive the EA's latest MT5 deals export under an auto-incremented name
    (cfg_001.csv, cfg_002.csv, ...), one per backtest, ready for the PBO test:
        python demo\collect_pbo.py "<Dest folder>"

.DESCRIPTION
    The EA always writes the SAME file (Common\Files\EGP_deals.csv); every backtest
    OVERWRITES the previous one, and the MT5 optimizer truncates it at each pass.
    Run this script ONCE AFTER EACH single backtest to copy that run out under a
    distinct name into a collection folder. When the folder holds >= 2 (ideally 10+)
    distinct configs, point collect_pbo.py at it.

.PARAMETER Dest
    Folder collecting one CSV per config. Created if missing. Default: .\pbo_configs

.PARAMETER Source
    Path to the EA's deals export. Default: the standard MT5 Common\Files location
    ($env:APPDATA\MetaQuotes\Terminal\Common\Files\EGP_deals.csv).

.PARAMETER Name
    Explicit output name (e.g. baseline). If omitted, auto-increments <Prefix>NNN.

.PARAMETER Prefix
    Prefix for auto-naming. Default: cfg_

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File demo\save_config_deals.ps1
    # copies Common\Files\EGP_deals.csv to .\pbo_configs\cfg_001.csv (then 002, 003, ...)

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File demo\save_config_deals.ps1 -Dest H:\pbo_configs -Name baseline
#>
[CmdletBinding()]
param(
    [string]$Dest   = (Join-Path (Get-Location) 'pbo_configs'),
    [string]$Source = (Join-Path $env:APPDATA 'MetaQuotes\Terminal\Common\Files\EGP_deals.csv'),
    [string]$Name   = '',
    [string]$Prefix = 'cfg_'
)

if (-not (Test-Path -LiteralPath $Source)) {
    Write-Error ("Deals export not found: {0}`n  Run ONE backtest first (the EA writes it via EGP_MHO_DealsExport.mqh), or pass -Source <path>." -f $Source)
    exit 1
}

if (-not (Test-Path -LiteralPath $Dest)) {
    New-Item -ItemType Directory -Force -Path $Dest | Out-Null
}

if ([string]::IsNullOrWhiteSpace($Name)) {
    # auto-increment: find the max NNN among <Prefix>NNN.csv, add 1
    $rx  = [regex]('^' + [regex]::Escape($Prefix) + '(\d+)\.csv$')
    $max = 0
    Get-ChildItem -LiteralPath $Dest -Filter ($Prefix + '*.csv') -File -ErrorAction SilentlyContinue | ForEach-Object {
        $m = $rx.Match($_.Name)
        if ($m.Success) {
            $n = [int]$m.Groups[1].Value
            if ($n -gt $max) { $max = $n }
        }
    }
    $fileName = '{0}{1:000}.csv' -f $Prefix, ($max + 1)
} else {
    $fileName = $Name
    if ($fileName -notmatch '\.csv$') { $fileName += '.csv' }
}

$target = Join-Path $Dest $fileName
if (Test-Path -LiteralPath $target) {
    Write-Error ("Target already exists: {0}  (pass a different -Name, or clear the folder)." -f $target)
    exit 1
}

Copy-Item -LiteralPath $Source -Destination $target
$srcInfo = Get-Item -LiteralPath $Source
$count   = (Get-ChildItem -LiteralPath $Dest -Filter '*.csv' -File).Count

Write-Host ('[saved]  {0}  <-  EGP_deals.csv (exported {1})' -f $target, $srcInfo.LastWriteTime)
Write-Host ('[folder] {0}  now holds {1} CSV(s)' -f $Dest, $count)
if ($count -ge 2) {
    Write-Host ('[next]   python demo\collect_pbo.py "{0}"' -f $Dest)
} else {
    Write-Host '[next]   run another config''s backtest, then re-run this script (PBO needs >= 2 configs).'
}
