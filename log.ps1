param (
    [string]$date = "20241006"
)
$remoteUser = "sqa24"
# $remoteUser = "dcao2028" # remember to change this

if ($remoteUser -eq "sqa24") {
    $key = "C:\Users\14695\.ssh\ubuntu\id_rsa"
} else {
    $key = "C:\Users\14695\.ssh\ubuntu\satori_rsa_common"
}

$remoteHost = "satori-login-002.mit.edu"
$localdir = "C:\Users\14695\Desktop\group\personal_experiments\diffusion\log"

scp -i $key $remoteUser@${remoteHost}:$remoteDir/${date}* ${localdir}