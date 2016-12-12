#!/bin/sh

rm -r /var/tmp/encrypted
mkdir -p /var/tmp/encrypted
cd ~/Dropbox/phone_data/Sensor\ Research
for BEFORE in 0
do
    DAY=`python -c "import datetime; d = datetime.datetime.now() - datetime.timedelta(${BEFORE}); print d.strftime('%Y_%m_%d')"`
    cp *_${DAY}_* /var/tmp/encrypted/
done

rm -r /var/tmp/decrypted
mkdir -p /var/tmp/decrypted

KEY=~/phone-theft-research/dashboard-scripts/ucb_keypair/ucb.privatekey
cd ~/phone-theft-research/classifier-scripts/
python bintocsv.py $KEY -s /var/tmp/encrypted -d /var/tmp/decrypted/
python ClassifierLog.py
