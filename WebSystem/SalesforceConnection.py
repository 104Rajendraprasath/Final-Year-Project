from simple_salesforce import Salesforce
import requests
SF_ORG_ALIAS = "MyProjectOrg"
SF_LOGIN_URL="https://uce3-dev-ed.develop.my.salesforce.com/services/oauth2/token"
try:
        print("[SALESFORCE] Authenticating with OAuth...")

        payload = {
            "grant_type": "client_credentials",
            
          }

        auth_response = requests.post(SF_LOGIN_URL, data=payload)
        

        print("STATUS CODE:", auth_response.status_code)
        print("RAW RESPONSE:", auth_response.text)

        auth_data = auth_response.json()
        

        access_token = auth_data["access_token"]
        instance_url = auth_data["instance_url"]

        sf = Salesforce(instance_url=instance_url, session_id=access_token)

        sf.Security_Alert__c.create({
            "Camera_ID__c": camera_id,
            "Location__c": location,
            "Confidence__c": float(confidence * 100),
            "Status__c": "New",
            "Message__c": threat_type
        })

        print("[SUCCESS] Salesforce Record Created")
        

except Exception as e:
        print("[SALESFORCE ERROR]", e)
       
