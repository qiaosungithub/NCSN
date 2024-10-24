# init params
param (
    [string]$f="*.py"
)

$remoteUser = "sqa24"
# $remoteUser = "dcao2028" # remember to change this
# $remoteUser = "jqdai"

if ($remoteUser -eq "sqa24") {
    $key = "C:\Users\14695\.ssh\ubuntu\id_rsa"
} else {
    $key = "C:\Users\14695\.ssh\ubuntu\satori_rsa_common"
}

$remoteHost = "satori-login-002.mit.edu"
$remoteDir = "NCSN" # remember to change this

# 遍历当前目录下的所有文件
Get-ChildItem -File -Filter $f | ForEach-Object {
# Get-ChildItem -File | ForEach-Object {

    $localFile = $_.FullName
    $remoteFile = "$remoteUser@${remoteHost}:$remoteDir/$_"

    # 使用 scp 命令上传文件
    scp -i $key $localFile $remoteFile
}

# to download a file from remote server

# $remoteUser
# $remoteHost
# $remoteDir

# $remoteFile = "$remoteUser@${remoteHost}:$remoteDir/ebm_hamiltonian/CIFAR/data/ckpt.pth"
# $localFile = "C:\Users\14695\Desktop\group\personal_experiments\ebm_hamiltonian\CIFAR\data\ckpt.pth"

# scp -i $key $remoteFile $localFile

# scp -i C:\Users\14695\.ssh\ubuntu\satori_rsa_common dcao2028@satori-login-002.mit.edu: C:\Users\14695\Desktop\group\personal_experiments\ViT\ViT_vanilla\logs