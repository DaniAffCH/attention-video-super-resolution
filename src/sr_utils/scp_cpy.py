import subprocess
import os
import time
key = "spqr"
path = "~/Desktop/daniele_stuff"
ip = "192.168.0.42"
remote_user = "spqrnao"
file = "../images_inf/*.png"

update = 10

while True:
	bash_command = f"sshpass -p {key} scp {file} {remote_user}@{ip}:{path}"
	subprocess.run(bash_command, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
	bash_command = f"rm {file}"
	subprocess.run(bash_command, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
	print(f"[COPYFILE] PID = {os.getpid()}")
	time.sleep(update)

