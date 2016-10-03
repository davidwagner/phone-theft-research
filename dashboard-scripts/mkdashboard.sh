#!/bin/sh
cd ~/phone-theft-research/dashboard-scripts/
rm -r /var/tmp/decrypted
mkdir -p /var/tmp/decrypted
python bintocsv.py ucb_keypair/ucb.privatekey -s ~/Dropbox/phone_data/Sensor\ Research -d /var/tmp/decrypted/
python logchecker.py /var/tmp/decrypted
