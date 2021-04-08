from boxsdk import DevelopmentClient
from boxsdk import OAuth2, Client
from boxsdk import JWTAuth


import os
# from boxsdk.exception import BoxAPIException
# from boxsdk.network.logging_network import LoggingNetwork
# from boxsdk.object.collaboration import CollaborationRole
# client = DevelopmentClient()
# config = JWTAuth.from_settings_file('config.json')
auth_token = "qshrgQGBHcwWWuWFBK8Y9PpFUqcaSv85"
config = OAuth2(
    client_id='qgbl5vbwefkfx8h0rfudledazlkpp2c0',
    client_secret='cEcc6w1PIOchMTSydiaUWPqhlEP2rpYL',
    access_token='f{auth_token}',
)
client = Client(config)

## current user

# upload file
def upload(binary, token):
    """
    uploads to dl_binaries folderkkkkkkkkkkk
    both imputs instrings...
    """
    global auth_token
    auth_token = token
    user = client.user().get()
    print('The current user ID is {0}'.format(user.id))
    root_folder = client.folder(folder_id='0').get() # get root folder
    print('The root folder is owned by: {0}'.format(root_folder.owned_by['login']))
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'dl_binaries/{binary}')
    a_file = root_folder.upload(file_path, file_name=binary)
    print('{0} uploaded: '.format(a_file.get()['name']))

# get items in folder
# items = root_folder.get_items(limit=100, offset=0)
# print('This is the first 100 items in the root folder:')
# for item in items:
#   print("   " + item.name)
        
# # delete a folder
# client.make_request('DELETE', 'https://api.box.com/2.0/folders/FOLDER_ID?recursive=true')


# # create new folder
# r = client.make_request(
#     'POST',
#     'https://api.box.com/2.0/folders',
#     data= json.dumps({'name': 'hoangdev','parent':{"id": "0"}})
# ).json()
# print r['id']
"""
	Files
"""
'''
# Upload a file to Box!
from StringIO import StringIO

stream = StringIO()
stream.write('Box Python SDK test!')
stream.seek(0)
box_file = client.folder('0').upload_stream(stream, 'box-python-sdk-test.txt')
print box_file.name
'''

# # delete file
# print a_file.delete()

# # rename file
# bar = a_file.rename('bar.txt')
# print('Rename succeeded: {0}'.format(bool(bar)))

## update file with other file
# file_v1 = root_folder.upload(file_path, file_name='file_v1.txt')
# # print 'File content after upload: {}'.format(file_v1.content())
# file_v2_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'file_v2.txt')
# file_v2 = file_v1.update_contents(file_v2_path)
# print 'File content after update: {}'.format(file_v2.content())
