'''
http://opensource.box.com/box-python-sdk/tutorials/intro.html
https://github.com/box/box-python-sdk/blob/1.5/demo/example.py
https://github.com/box/box-python-sdk/blob/v2.0.0/demo/example.py
'''
from boxsdk import DevelopmentClient
from boxsdk import OAuth2, Client
from boxsdk.exception import BoxAPIException
from boxsdk.network.logging_network import LoggingNetwork
from boxsdk.object.collaboration import CollaborationRole
#client = DevelopmentClient()

auth = OAuth2(
    client_id='qgbl5vbwefkfx8h0rfudledazlkpp2c0',
    client_secret='cEcc6w1PIOchMTSydiaUWPqhlEP2rpYL',
    access_token='wJaJr4QXlA94JPZ7okNXTGdNrHmBjjxW',
)
client = Client(auth)

## current user
user = client.user().get()
print('The current user ID is {0}'.format(user.id))

# me = client.user(user_id='me').get(fields=['login'])
# print('The email of the user is: {0}'.format(me['login']))

## Making API Calls Manually
# use these endpoints by using the make_request method of the Client
# https://box-content.readme.io/reference#get-metadata-schema
# # Returns a Python dictionary containing the result of the API request
# json_response = client.make_request(
#     'GET',
#     client.get_url('metadata_templates', 'enterprise', 'customer', 'schema'),
# ).json()

# ## if require a body
# # JSONify the body
# body = json.dumps({"is_accepted":true})

# # Pass body as "data" argument
# client.make_request(method, url, data = body)

# ## logging
# from boxsdk import LoggingClient
# client = LoggingClient()
# client.user().get()

# Use a custom logger
#client = Client(oauth, network_layer=LoggingNetwork(logger))

"""
	Folder
"""
## create new folder
# r = client.make_request(
#     'POST',
#     'https://api.box.com/2.0/folders',
#     data= json.dumps({'name': 'hoangdev','parent':{"id": "0"}})
# ).json()
# print r['id']

# upload a folder
#https://box-content.readme.io/reference#update-information-about-a-folder

# create subfolder
# collab_folder = root_folder.create_subfolder('collab folder')
# print('Folder {0} created'.format(collab_folder.get()['name']))
# # delete folder
# collab_folder.delete()

# rename folder
# bar = collab_folder.rename('bar')
# print('Renamed to {0}'.format(bar.get()['name']))

# # add collaboration
# collaboration = collab_folder.add_collaborator('someone@example.com', CollaborationRole.VIEWER)
# print('Created a collaboration')
# # edit role
# modified_collaboration = collaboration.update_info(role=CollaborationRole.EDITOR)
# print('Modified a collaboration: {0}'.format(modified_collaboration.role))
# collaboration.delete()	# Deleted a collaboration

# # get share folder link
# shared_link = collab_folder.get_shared_link()
# print('Got shared link:' + shared_link)

# get root folder
root_folder = client.folder(folder_id='0').get()
print('The root folder is owned by: {0}'.format(root_folder.owned_by['login']))

# get items in folder
# items = root_folder.get_items(limit=100, offset=0)
# print('This is the first 100 items in the root folder:')
# for item in items:
#   print("   " + item.name)
        
# # delete a folder
# client.make_request('DELETE', 'https://api.box.com/2.0/folders/FOLDER_ID?recursive=true')

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

# upload file
file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'file.txt')
a_file = root_folder.upload(file_path, file_name='i-am-a-file.txt')
print('{0} uploaded: '.format(a_file.get()['name']))

# # delete file
# print a_file.delete()

# # rename file
# bar = a_file.rename('bar.txt')
# print('Rename succeeded: {0}'.format(bool(bar)))

## update file with other file
file_v1 = root_folder.upload(file_path, file_name='file_v1.txt')
# print 'File content after upload: {}'.format(file_v1.content())
file_v2_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'file_v2.txt')
file_v2 = file_v1.update_contents(file_v2_path)
# print 'File content after update: {}'.format(file_v2.content())

# ## search files
# search_results = client.search().query(
#   'i-am-a-file.txt',
#   limit=2,
#   offset=0,
#   ancestor_folders=[client.folder(folder_id='0')],
#   file_extensions=['txt'],
# )
# for item in search_results:
#   item_with_name = item.get(fields=['name'])
#   print('matching item: ' + item_with_name.id)
#   else:
#     print('no matching items')

    