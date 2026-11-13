# Empêche la mise en veille tant que la chaîne d'entraînement tourne (ne modifie aucun réglage Windows).
$sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$api = Add-Type -MemberDefinition $sig -Name Power -Namespace Win32 -PassThru
# ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
$null = $api::SetThreadExecutionState([uint32]"0x80000003")
$log = Join-Path $PSScriptRoot "chain.log"
while (-not (Select-String -Path $log -Pattern "CHAINE TERMINEE|ECHEC" -Quiet)) { Start-Sleep -Seconds 60 }
$null = $api::SetThreadExecutionState([uint32]"0x80000000")
"keep_awake termine"
