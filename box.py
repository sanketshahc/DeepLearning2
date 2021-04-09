from boxsdk import OAuth2, Client

import os
# from boxsdk.exception import BoxAPIException
# from boxsdk.network.logging_network import LoggingNetwork
# from boxsdk.object.collaboration import CollaborationRole
# client = DevelopmentClient()
# config = JWTAuth.from_settings_file('config.json')
oauth = OAuth2(
    client_id='df6mdynv4n5hkni08jb4zufbmki6r2rc',
    client_secret='4ZQMQ1AAYxNMzpBB7KKqaQWchEXtdtLU',
    access_token='shsRXKdhsXNTsHP6Ljc2IG9mNuojX4dM',
    refresh_token='Im0STIkgCJcPn5bfvWUEain6czRTDeh5stsdUNXEcrTB63B1tNBh855Rl0qxScFi'
)

# assert len(oauth) == 4
client = Client(oauth)

# upload file
def upload(filename):
    """
    input string filename.
    uploads to dl_binaries folderkkkkkkkkkkk
    both imputs instrings...
    """

    user = client.user().get()
    print('The current user ID is {0}'.format(user.id))
    root_folder = client.folder(folder_id='0').get() # get root folder
    print('The root folder is owned by: {0}'.format(root_folder.owned_by['login']))
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'pickled_binaries/{filename}')
    print('path',file_path)
    a_file = root_folder.upload(file_path, file_name=filename)
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
