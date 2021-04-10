from boxsdk import OAuth2, Client
import requests
import os
# from boxsdk.exception import BoxAPIException
# from boxsdk.network.logging_network import LoggingNetwork
# from boxsdk.object.collaboration import CollaborationRole
# client = DevelopmentClient()
# config = JWTAuth.from_settings_file('config.json')
client_id='df6mdynv4n5hkni08jb4zufbmki6r2rc'
client_secret='4ZQMQ1AAYxNMzpBB7KKqaQWchEXtdtLU'
access_token=None
refresh_token=None
with open("./auth.txt", "r") as f:
    access_token = f.readline()
    refresh_token = f.readline()

params = {
    "client_id":client_id, 
    "client_secret":client_secret, 
    "refresh_token":refresh_token,
    "grant_type":"refresh_token"
    }
    
# upload file
def upload(filename):
    """
    input string filename.
    uploads to dl_binaries folderkkkkkkkkkkk
    both imputs instrings...
    """
    global access_token
    global refresh_token

    print('Requesting Refresh Token.')
    r = requests.post("https://api.box.com/oauth2/token",params)
    r = r.json()
    print(r)
    access_token = r["access_token"]
    refresh_token = r["refresh_token"]
    with open('./auth.txt',"w") as auth:
        auth.write(access_token)
        auth.write('\n')
        auth.write(refresh_token)
    print('Tokens Updated')
    oauth = OAuth2(
        client_id=client_id,
        client_secret=client_secret,
        access_token=access_token,
        refresh_token=refresh_token
        )
    client = Client(oauth)
    user = client.user().get()
    print(user, "inititated")
    print('The current user ID is {0}'.format(user.id))

    try:
        root_folder = client.folder(folder_id='0').get() # get root folder
        print('The root folder is owned by: {0}'.format(root_folder.owned_by['login']))
        file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'pickled_binaries/{filename}')
        print('path',file_path)
        a_file = root_folder.upload(file_path, file_name=filename)
        print('{0} uploaded: '.format(a_file.get()['name']))
    except Exception:
        print("upload failed....")

