import json
import requests

url = 'http://127.0.0.1:8002/feeling_pred'

input_data_for_model = {

    'StringInput' : "I feel so alone. Every day is a struggle to get out of bed. Nothing brings me joy anymore, and I can't shake this overwhelming sense of sadness. I feel like I'm drowning, and I don't know how to ask for help. It's like I'm stuck in this endless cycle of despair, and I don't see a way out."






}

input_json = json.dumps(input_data_for_model)
response = requests.post(url, data = input_json)

# Check if the response is successful (status code 200)
if response.status_code == 200:
    response_text = str(response.text.strip().strip('"'))  # Remove leading/trailing quotes
else:
    print(f"Error: {response.status_code} {response.text}")    

# Split response text into lines using regular expressions
response_lines = response.text.strip().split('\\n')
print(response_lines)
# Iterate through response lines
flag = None
for line in response_lines:
    print(line)
    if ':' in line:
        key, value = line.split(':', 1)  # Split each line into key-value pair
        if value.strip() == 'True':
            flag = key.strip()
            break

# Create content based on flag
if flag:
    content = {
        "name": "",
        "message": f"This message has been censored due to being {flag}."
    }
else:       
    content = {
        "name": '',
        "message": ""
    }
    
print("Message")