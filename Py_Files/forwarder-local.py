import paho.mqtt.client as mqtt


LOCAL_MQTT_HOST="localhost"
REMOTE_MQTT_HOST="18.208.212.20"
LOCAL_MQTT_PORT=1883
REMOTE_MQTT_PORT=30206
LOCAL_MQTT_TOPIC="audio_file"
REMOTE_MQTT_TOPIC="audio_file"

def on_connect_local(client, userdata, flags, rc):
        print("connected to local broker with rc: " + str(rc))
        client.subscribe(LOCAL_MQTT_TOPIC)
	
def on_message(client,userdata, msg):
  try:
    print("message received: ",msg.payload.decode)
    msg = msg.payload
    remote_mqttclient.publish(REMOTE_MQTT_TOPIC, payload=msg, qos=0, retain=False)
  except:
    print("Unexpected error")

local_mqttclient = mqtt.Client()
local_mqttclient.on_connect = on_connect_local
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
local_mqttclient.on_message = on_message

remote_mqttclient = mqtt.Client()
remote_mqttclient.on_connect = on_connect_local
remote_mqttclient.connect(REMOTE_MQTT_HOST, REMOTE_MQTT_PORT, 60)

# go into a loop
local_mqttclient.loop_forever()
remote_mqttclient.loop_forever()
