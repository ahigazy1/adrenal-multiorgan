# Windows: keep the Colab runtime talked to (colab_compare.sh poll) and keep this PC awake while it exists.
# Colab reclaims a runtime that no client talks to, and a sleeping PC talks to nothing.
#     powershell -ExecutionPolicy Bypass -File colab_poll.ps1
# Only idle sleep is blocked, for as long as the run lasts; closing the lid or choosing Sleep still sleeps.
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);'
[Win32.Power]::SetThreadExecutionState(0x80000001) | Out-Null  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
wsl -d Ubuntu -- bash -lc "bash /mnt/c/Users/ahiga/Desktop/adrenal-multiorgan/colab_compare.sh poll"
[Win32.Power]::SetThreadExecutionState(0x80000000) | Out-Null  # the run is over: sleep is allowed again
