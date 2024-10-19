import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_mongo_client():
    # Get the MongoDB connection string from the environment variable
    mongo_uri = os.getenv("MONGODB_URI")

    if not mongo_uri:
        raise ValueError("Error: MONGODB_URI not found. Please check your .env file.")
    
    # Create and return a MongoClient with TLS settings
    try:
        client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
        print("Connected to MongoDB successfully.")
        return client
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        raise e