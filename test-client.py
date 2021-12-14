# client
import json

import requests
import simplejson

signal = b'AAAAAAAgrD8AAAAAAMawPwAAAAAAqLA/AAAAAABMqz8AAAAAAEikPwAAAAAAoKA/AAAAAAAcoT8AAAAAADyhPwAAAAAAAJk/'
# signal_json = signal.decode('utf8').replace("'", '"')
# data = json.loads(signal_json)
# data_send = json.dumps(data)
# data = {'data': data_send}
data = {'data': signal}
data_json = simplejson.dumps(data)
# payload = {'json_payload': data_json}
r = requests.post("http://127.0.0.1:5000/", json=data_json)
print(r.status_code)
print(r.json())
