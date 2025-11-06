from pymongo import MongoClient
import certifi


uri = "mongodb+srv://uishivansh2503_db_user:3svnnxBJ04eu4iFb@kajkarma.wapdfls.mongodb.net/?appName=Kajkarma"
client = MongoClient(uri, tls=True, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=30000)
print(client.server_info())  # should print a dict
