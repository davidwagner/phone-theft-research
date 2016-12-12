#!/bin/sh

rm -r /var/tmp/encrypted2
mkdir -p /var/tmp/encrypted2
cd ~/Dropbox/phone_data/Sensor\ Research
for BEFORE in 0
do
    DAY=`python -c "import datetime; d = datetime.datetime.now() - datetime.timedelta(${BEFORE}); print d.strftime('%Y_%m_%d')"`
    cp *_${DAY}_* /var/tmp/encrypted2
done

rm -r /var/tmp/decrypted2
mkdir -p /var/tmp/decrypted2

cd ~/phone-theft-research/dashboard-scripts/
nice -19 python bintocsv.py ucb_keypair/ucb.privatekey -s /var/tmp/encrypted2 -d /var/tmp/decrypted2/
cd ~/phone-theft-research/classifiers-scripts
nice -19 python ClassifierLog.py
