import json
import time
import uuid
import os

print("GUEST BOOKING INTERFACE")


guest_name = str(input("Guest name:"))
print("You entered " + str(guest))

date = str(input("Visit Date: (YYYYMMDD)"))
if not date:
	date = time.strftime("%Y%m%d")
print(str(date))

#Load object

g_pass = {}
g_pass['_id'] = ''
g_pass['name'] = guest_name
t = uuid.uuid4().hex
g_pass['token'] = t[:5]
g_pass['date'] = date
g_pass['time'] = ''
g_pass['guest_type'] = 'tenant_guest'
g_pass['face_id'] = ''
g_pass['created'] = time.strftime("%Y%m%d-%H%M%S")
g_pass['host'] = ''


#Save File

pass_filename = 'g_pass/%s/pass_%s.json'%(g_pass['date'],g_pass['token'])

if not os.path.exists(pass_filename):
	os.makedirs(os.path.dirname(pass_filename), exist_ok=True)
	with open(pass_filename, "w") as f:
	    try:
	        print(g_pass)
	        f.write(json.dumps(g_pass))
	        print("Pass SAVED")
	    finally:
	        f.close()
