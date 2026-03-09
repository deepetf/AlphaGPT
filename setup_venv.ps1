param(
    [string]$VenvPath = ".venv",
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

function Find-PythonCommand {
    $candidates = @(
        @{ Command = "py"; Args = @("-3.10"); Label = "py -3.10" },
        @{ Command = "py"; Args = @("-3"); Label = "py -3" },
        @{ Command = "py"; Args = @(); Label = "py" },
        @{ Command = "python"; Args = @(); Label = "python" }
    )

    foreach ($candidate in $candidates) {
        if (-not (Get-Command $candidate.Command -ErrorAction SilentlyContinue)) {
            continue
        }

        try {
            & $candidate.Command @($candidate.Args + @("--version")) | Out-Null
            return $candidate
        }
        catch {
            continue
        }
    }

    throw "未找到可用的 Python 解释器。请先安装 Python 3.10+，或确认 py/python 已加入 PATH。"
}

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvFullPath = Join-Path $projectRoot $VenvPath
$requirementsPath = Join-Path $projectRoot "requirements.txt"

if (-not (Test-Path $requirementsPath)) {
    throw "未找到 requirements.txt: $requirementsPath"
}

$pythonSpec = Find-PythonCommand
Write-Host "使用解释器: $($pythonSpec.Label)" -ForegroundColor Cyan

if (-not (Test-Path $venvFullPath)) {
    Write-Host "创建虚拟环境: $venvFullPath" -ForegroundColor Cyan
    & $pythonSpec.Command @($pythonSpec.Args + @("-m", "venv", $venvFullPath))
}
else {
    Write-Host "虚拟环境已存在，跳过创建: $venvFullPath" -ForegroundColor Yellow
}

$venvPython = Join-Path $venvFullPath "Scripts\\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "创建后的虚拟环境缺少解释器: $venvPython"
}

Write-Host "升级 pip" -ForegroundColor Cyan
& $venvPython -m pip install --upgrade pip

if (-not $SkipInstall) {
    Write-Host "安装项目依赖: $requirementsPath" -ForegroundColor Cyan
    & $venvPython -m pip install -r $requirementsPath
}
else {
    Write-Host "已跳过依赖安装" -ForegroundColor Yellow
}

$activateScript = Join-Path $venvFullPath "Scripts\\Activate.ps1"

Write-Host ""
Write-Host "完成。后续可这样启用虚拟环境：" -ForegroundColor Green
Write-Host "Set-ExecutionPolicy -Scope Process Bypass"
Write-Host "& `"$activateScript`""
Write-Host ""
Write-Host "不激活也可直接运行：" -ForegroundColor Green
Write-Host "& `"$venvPython`" -m model_core.engine"
