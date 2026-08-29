# Creates a Start Menu shortcut for Orbit, so it shows up in Windows Search
# and launches like a normal installed app instead of needing a terminal.
# Uses pythonw.exe (the windowless launcher every CPython install ships
# alongside python.exe) instead of python.exe, so launching it doesn't pop
# up a console window behind the floating avatar -- the one other thing
# that made this look like "a script" rather than "an app" day to day.
#
# Safe to re-run any time (e.g. after moving the repo or switching which
# venv you use) -- it just overwrites the existing shortcut.
#
# This does NOT freeze the app into a standalone .exe (Task Manager will
# still show pythonw.exe as the process) -- see README's Known limitations
# for why that's a deliberately bigger, riskier step this project isn't
# taking yet. This just makes the existing script discoverable/launchable
# like an app, which is the part that was actually missing.

$here = Split-Path -Parent $MyInvocation.MyCommand.Path

# Prefer a local venv over a bare system Python, checked in this order --
# first match wins. Add your own venv folder name here if it's not one of
# these three.
$venvCandidates = @(".venv", "venv311", "venv")
$pythonw = $null
foreach ($v in $venvCandidates) {
    $candidate = Join-Path $here "$v\Scripts\pythonw.exe"
    if (Test-Path $candidate) {
        $pythonw = $candidate
        break
    }
}
if (-not $pythonw) {
    $cmd = Get-Command pythonw.exe -ErrorAction SilentlyContinue
    if ($cmd) { $pythonw = $cmd.Source }
}
if (-not $pythonw) {
    Write-Error "Couldn't find pythonw.exe in .venv/venv311/venv or on PATH. Set up your Python environment first (see README's Setup section), then re-run this script."
    exit 1
}

$startMenu = [Environment]::GetFolderPath("Programs")
$shortcutPath = Join-Path $startMenu "Orbit.lnk"

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $pythonw
$shortcut.Arguments = '"' + (Join-Path $here "assistant_app.py") + '"'
# assistant_app.py resolves piper_models/ (tts.PIPER_MODEL) relative to the
# CURRENT WORKING DIRECTORY, not its own script location -- WorkingDirectory
# is what makes that resolve correctly when launched from the Start Menu
# instead of a terminal already sitting in this folder.
$shortcut.WorkingDirectory = $here
$shortcut.IconLocation = Join-Path $here "icon.ico"
$shortcut.Description = "Orbit -- local voice assistant client"
$shortcut.Save()

Write-Output "Created Start Menu shortcut: $shortcutPath"
Write-Output "Using interpreter: $pythonw"
Write-Output "Search for 'Orbit' in the Start Menu / Windows Search to launch it."
