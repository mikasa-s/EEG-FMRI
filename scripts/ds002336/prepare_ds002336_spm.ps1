param(
    [string]$DsRoot = "../ds002336",
    [string]$OutputRoot = "cache/ds002336",
    [string[]]$Subjects = @(),
    [string[]]$Tasks = @("motorloc", "MIpre", "MIpost", "eegNF", "fmriNF", "eegfmriNF"),
    [ValidateSet("none", "subject", "loso")]
    [string]$SplitMode = "loso",
    [int]$TrainSubjects = 7,
    [int]$ValSubjects = 2,
    [int]$TestSubjects = 1,
    [int]$EegSeqLen = 20,
    [int]$EegPatchLen = 200,
    [bool]$DropEcg = $true,
    [ValidateRange(1, 32)]
    [int]$ParallelJobs = 2
)

$ErrorActionPreference = "Stop"

function Convert-ToMatlabCellArray {
    param([string[]]$Values)

    if (-not $Values -or $Values.Count -eq 0) {
        return "{}"
    }

    $escaped = foreach ($Value in $Values) {
        "'" + ($Value -replace "'", "''") + "'"
    }
    return "{" + ($escaped -join ",") + "}"
}

function Get-SubjectList {
    param(
        [string]$DatasetRoot,
        [string[]]$RequestedSubjects
    )

    if ($RequestedSubjects.Count -gt 0) {
        return $RequestedSubjects
    }

    return @(Get-ChildItem -Path $DatasetRoot -Directory -Filter "sub-*" | Sort-Object Name | ForEach-Object { $_.Name })
}

function Start-SpmMatlabProcess {
    param(
        [string]$MatlabScriptDir,
        [string[]]$SubjectBatch,
        [string[]]$TaskList,
        [string]$LogDir
    )

    $subjectExpr = Convert-ToMatlabCellArray $SubjectBatch
    $taskExpr = Convert-ToMatlabCellArray ($TaskList | ForEach-Object { "task-$_" })
    $matlabCommand = "addpath('$MatlabScriptDir'); run_spm_preproc_xp1($subjectExpr,$taskExpr);"
    $safeName = (($SubjectBatch -join "_") -replace "[^A-Za-z0-9_-]", "_")
    $stdoutPath = Join-Path $LogDir ($safeName + ".log")
    $stderrPath = Join-Path $LogDir ($safeName + ".err.log")
    $launcherPath = Join-Path $LogDir ($safeName + ".launcher.ps1")
    $launcherLines = @(
        '$ErrorActionPreference = ''Stop''',
        ('matlab -batch "' + $matlabCommand.Replace('"', '""') + '"'),
        'exit $LASTEXITCODE'
    )
    Set-Content -Path $launcherPath -Value $launcherLines -Encoding UTF8
    $process = Start-Process -FilePath "powershell" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $launcherPath) -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath

    return [pscustomobject]@{
        Subjects = $SubjectBatch
        Process  = $process
        StdOut   = $stdoutPath
        StdErr   = $stderrPath
        Launcher = $launcherPath
    }
}

function Complete-SpmJobs {
    param(
        [System.Collections.Generic.List[object]]$Jobs
    )

    foreach ($job in @($Jobs)) {
        if (-not $job.Process.HasExited) {
            continue
        }

        $job.Process.WaitForExit()
        $job.Process.Refresh()

        $stdoutTail = @()
        $stderrTail = @()
        if (Test-Path $job.StdOut) {
            $stdoutTail = @(Get-Content $job.StdOut -Tail 40)
        }
        if (Test-Path $job.StdErr) {
            $stderrTail = @(Get-Content $job.StdErr -Tail 40)
        }

        $exitCode = $job.Process.ExitCode
        if ($null -eq $exitCode -or ("$exitCode").Trim() -eq "") {
            if ($stderrTail.Count -eq 0 -and ($stdoutTail -join "`n") -match "All requested ds002336 preprocessing jobs finished") {
                $exitCode = 0
            }
        }

        Write-Host ("Completed SPM job for: " + ($job.Subjects -join ", ") + " (exit " + $exitCode + ")")
        [void]$Jobs.Remove($job)

        if ($exitCode -ne 0) {
            if ($stdoutTail.Count -gt 0) {
                Write-Host "Last stdout lines:"
                $stdoutTail
            }
            if ($stderrTail.Count -gt 0) {
                Write-Host "Last stderr lines:"
                $stderrTail
            }
            throw ("SPM12 preprocessing failed for subjects: " + ($job.Subjects -join ", "))
        }
    }
}

function Invoke-SpmMatlabForeground {
    param(
        [string]$MatlabScriptDir,
        [string[]]$SubjectBatch,
        [string[]]$TaskList
    )

    $subjectExpr = Convert-ToMatlabCellArray $SubjectBatch
    $taskExpr = Convert-ToMatlabCellArray ($TaskList | ForEach-Object { "task-$_" })
    $matlabCommand = "addpath('$MatlabScriptDir'); run_spm_preproc_xp1($subjectExpr,$taskExpr);"

    matlab -batch $matlabCommand
    if ($LASTEXITCODE -ne 0) {
        throw ("SPM12 preprocessing failed for subjects: " + ($SubjectBatch -join ", "))
    }
}

$workspaceRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$matlabScriptDir = Resolve-Path (Join-Path $workspaceRoot "preprocess")
$logDir = Join-Path $workspaceRoot "outputs\logs\ds002336_spm"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

Set-Location $workspaceRoot
$prepareScript = Resolve-Path (Join-Path $PSScriptRoot "prepare_ds002336.ps1")
$subjectList = Get-SubjectList -DatasetRoot $DsRoot -RequestedSubjects $Subjects

if ($subjectList.Count -eq 0) {
    throw "No ds002336 subjects were found to preprocess."
}

Write-Host "Running SPM12 preprocessing for ds002336..."
Write-Host ("Tasks: " + ($Tasks -join ", "))
Write-Host ("MATLAB script dir: " + $matlabScriptDir)
Write-Host ("Parallel jobs: " + $ParallelJobs)
Write-Host ("Subject count: " + $subjectList.Count)
if ($subjectList.Count -gt 0) {
    Write-Host ("Subjects: " + ($subjectList -join ", "))
}

if ($ParallelJobs -le 1 -or $subjectList.Count -le 1) {
    Invoke-SpmMatlabForeground -MatlabScriptDir $matlabScriptDir -SubjectBatch $subjectList -TaskList $Tasks
}
else {
    $jobs = [System.Collections.Generic.List[object]]::new()
    foreach ($subject in $subjectList) {
        while ($jobs.Count -ge $ParallelJobs) {
            Start-Sleep -Seconds 5
            Complete-SpmJobs -Jobs $jobs
        }

        Write-Host ("Starting SPM job for: " + $subject)
        $job = Start-SpmMatlabProcess -MatlabScriptDir $matlabScriptDir -SubjectBatch @($subject) -TaskList $Tasks -LogDir $logDir
        $jobs.Add($job)
    }

    while ($jobs.Count -gt 0) {
        Start-Sleep -Seconds 5
        Complete-SpmJobs -Jobs $jobs
    }
}

Write-Host "Running Python dataset preparation using SPM-preprocessed fMRI..."

$prepareArgs = @{
    DsRoot        = $DsRoot
    OutputRoot    = $OutputRoot
    Tasks         = $Tasks
    SplitMode     = $SplitMode
    TrainSubjects = $TrainSubjects
    ValSubjects   = $ValSubjects
    TestSubjects  = $TestSubjects
    EegSeqLen     = $EegSeqLen
    EegPatchLen   = $EegPatchLen
    DropEcg       = $DropEcg
    FmriSource    = "spm_smoothed"
}

if ($Subjects.Count -gt 0) {
    $prepareArgs.Subjects = $Subjects
}

& $prepareScript @prepareArgs