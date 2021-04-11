from boxsdk import OAuth2, Client

auth_token = "qshrgQGBHcwWWuWFBK8Y9PpFUqcaSv85"
oauth = OAuth2(
    client_id='df6mdynv4n5hkni08jb4zufbmki6r2rc',
    client_secret='4ZQMQ1AAYxNMzpBB7KKqaQWchEXtdtLU',
    access_token='shsRXKdhsXNTsHP6Ljc2IG9mNuojX4dM',
    refresh_token='Im0STIkgCJcPn5bfvWUEain6czRTDeh5stsdUNXEcrTB63B1tNBh855Rl0qxScFi'
)
auth_url, csrf_token = oauth.get_authorization_url('https://app.box.com')
# code = 'n14AHNkvcfTak3tPF9kmpG3kWufvdED6'
auth = open('./auth.txt',"w")
auth.write(auth_url)
auth.write('\n')
auth.close()
print(auth_url,'\nDODDDDDD')
# print('token',csrf_token)
x = input("auth url code?")
# code = 'n14AHNkvcfTak3tPF9kmpG3kWufvdED6'
access_token, refresh_token = oauth.authenticate(x)
# oauth['refresh_token'] = refresh_token
# oauth['access_token'] = access_token
print(access_token)
print(refresh_token)
auth = open('./auth.txt',"w")
auth.write(access_token)
auth.write('\n')
auth.write(refresh_token)
auth.close()

## Find AUth Code by going to developer view in chrome window, network tab, copy first url in log...