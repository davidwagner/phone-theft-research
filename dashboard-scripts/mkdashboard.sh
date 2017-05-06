#!/bin/sh

rm -r /var/tmp/encrypted
mkdir -p /var/tmp/encrypted
cd ~/Dropbox/phone_data/Sensor\ Research
for BEFORE in 0 1 2
do
    DAY=`python -c "import datetime; d = datetime.datetime.now() - datetime.timedelta(${BEFORE}); print d.strftime('%Y_%m_%d')"`
    cp *_{BleHrm,BatchedAccelerometer,TriggeredBleConnectedDevices}_${DAY}_* /var/tmp/encrypted/
done

rm -r /var/tmp/decrypted
mkdir -p /var/tmp/decrypted

cd ~/phone-theft-research/dashboard-scripts/
nice -19 python bintocsv.py ucb_keypair/ucb.privatekey -s /var/tmp/encrypted -d /var/tmp/decrypted/
nice -19 python logchecker.py /var/tmp/decrypted
