from pymongo import MongoClient
import certifi

client = MongoClient("mongodb+srv://uishivansh2503_db_user:3svnnxBJ04eu4iFb@kajkarma.wapdfls.mongodb.net/?appName=Kajkarma",
                     tls=True, tlsCAFile=certifi.where())
print(client.server_info())  # should print build info if OK
