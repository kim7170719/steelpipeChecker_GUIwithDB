@echo off
set updateCommand="curl https://api.dnsexit.com/dns/ud/?apikey=26Yr51YD8iF1K18VUkH9j37h8Fd37x -d host=steelpipenfu.run.place"
rem Create a scheduled task to run every 12 minutes. It must be run as System Administrator because it uses the SYSTEM account..
schtasks /create /tn "DNS Exit IP Update" /tr %updateCommand% /sc minute /mo 12 /ru SYSTEM
echo DNS Exit IP Update Task created.