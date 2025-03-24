from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import datetime
from dotenv import load_dotenv
import os

load_dotenv()
# Connecting to database.
uri = os.getenv('DATABASE_URL')
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["medicinal_plant"]
collection = db["history"]

# Insert new data record to the database.
def save_to_db(name,img,prediction):
    time=datetime.datetime.now()
    document = {'name':name,"image": img, "prediction": prediction,'time':time}
    collection.insert_one(document)