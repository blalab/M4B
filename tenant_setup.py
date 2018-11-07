import json
import time
import uuid
import os

print("TENANT ADMIN INTERFACE")


tenant_name = str(input("Tenant name:"))
print("You entered " + str(tenant_name))

date = time.strftime("%Y%m%d")

#Load object

t_pass = {}
t_pass['_id'] = ''
t_pass['name'] = tenant_name
t = uuid.uuid4().hex
t_pass['token'] = t[:5]
t_pass['date'] = date
t_pass['time'] = ''
t_pass['tenant_type'] = 'permanent'
t_pass['face_id'] = ''
t_pass['created'] = time.strftime("%Y%m%d-%H%M%S")



#Save File

pass_filename = 't_pass/%s/pass_%s.json'%(t_pass['date'],t_pass['token'])

if not os.path.exists(pass_filename):
	os.makedirs(os.path.dirname(pass_filename), exist_ok=True)
	with open(pass_filename, "w") as f:
	    try:
	        print(t_pass)
	        f.write(json.dumps(t_pass))
	        print("Pass SAVED")
	    finally:
	        f.close()
