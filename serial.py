import random 
string="qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM0123456789.!@#$^&"
serial_len=20

def generate_serial():
    serial=""
    for i in range(serial_len):
        serial+=random.choice(string)
    return serial
