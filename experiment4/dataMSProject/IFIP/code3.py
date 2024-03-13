import machine
import network
import time
from umqtt.simple import MQTTClient
import dht  # Import the DHT module

# Information for Wi-Fi connection
wifi_ssid = "XxX"
wifi_password = "Rgmb.4433,"

# MQTT Broker Information
mqtt_broker = "192.168.170.120"
mqtt_port = 1883  # Corrected variable name
mqtt_user = "username"
mqtt_passwd = "password"
mqtt_topic_motion = b"sensor_mo"
mqtt_topic_temperature = b"sensor_te"
mqtt_topic_humidity = b"sensor_hu"

# Define GPIO pins
motion_sensor_pin = machine.Pin(12, machine.Pin.IN)
dht_sensor_pin = machine.Pin(15)

# Connect to WI-FI
wlan = network.WLAN(network.STA_IF)
wlan.active(True)  # Corrected method name
wlan.connect(wifi_ssid, wifi_password)

while not wlan.isconnected():
    time.sleep(1)

print("Wi-Fi Connection Successful")

# Create MQTT Client
client = MQTTClient("pico", mqtt_broker, port=mqtt_port, user=mqtt_user, password=mqtt_password)

# Create DHT Object
dht_sensor = dht.DHT11(dht_sensor_pin)

while True:
    try:
        # Read data from the motion sensor
        motion_state = motion_sensor_pin.value()  # Removed extra equal sign

        # Send motion sensor data over MQTT
        client.connect()
        client.publish(mqtt_topic_motion, str(motion_state))
        client.disconnect()

        print("Motion Sensor State:", motion_state)

        # Read data from the DHT11 sensor
        dht_sensor.measure()
        temperature = dht_sensor.temperature()
        humidity = dht_sensor.humidity()

        # Send temperature data over MQTT
        client.connect()
        client.publish(mqtt_topic_temperature, str(temperature))  # Corrected variable name
        client.disconnect()

        print("Temperature:", temperature)

        # Send humidity data over MQTT
        client.connect()
        client.publish(mqtt_topic_humidity, str(humidity))  # Corrected variable name
        client.disconnect()

        print("Humidity:", humidity)

        time.sleep(10)  # 10 seconds waiting time
    except Exception as e:
        print("Error:", e)
        time.sleep(10)  # Wait for 10 seconds in case of an error

