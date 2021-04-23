Go to amazon AWS IOT, press get started,chose windows and python create a thing and download the zip file unzip + save as a directory on your locals,change the python file basicPubSub.py in connect_device_package/aws-iot-device-sdk-python/samples/basicPubSub and change to this python file.

open the terminal on your local computer, go to the directory 

use command for setup:
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

test communication with server:
.\start.ps1

to see full data sent to the server,go the test->Subscribe to a topic->type #->subscribe
