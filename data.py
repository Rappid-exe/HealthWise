import pandas as pd
import numpy as np
from db_connect import get_mongo_client

# Define categories and premium adjustments
categories = ['gym', 'fast_food', 'healthy_groceries', 'alcohol', 'supplements', 'organic_food', 'sports_equipment']
category_adjustments = {
    'gym': -5,
    'healthy_groceries': -10,
    'supplements': -7,
    'organic_food': -8,
    'sports_equipment': -6,
    'fast_food': 8,
    'alcohol': 12
}

def generate_data(num_records=10000):
    """
    Generates a DataFrame with random categories and amounts, and assigns premium adjustments.
    """
    # Generate random data for categories and amounts
    data = {
        'category': np.random.choice(categories, num_records),
        'amount': np.random.uniform(10, 500, num_records)
    }
    
    # Convert to DataFrame and assign premium adjustments based on category
    df = pd.DataFrame(data)
    df['premium_adjustment'] = df['category'].map(category_adjustments)
    
    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def insert_data_to_mongo(df):
    """
    Inserts the generated DataFrame into MongoDB in batches.
    """
    # Get MongoDB connection
    client = get_mongo_client()
    db = client['user']
    collection = db['HealthWise']

    # Batch size for insertions to MongoDB
    batch_size = 1000
    try:
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            collection.insert_many(batch.to_dict('records'))
        print("Data inserted successfully.")
    except Exception as e:
        print(f"Error during data insertion: {e}")
    finally:
        # Close the MongoDB connection
        client.close()
        print("MongoDB connection closed.")

if __name__ == '__main__':
    # Generate the data
    df = generate_data()

    # Insert the data into MongoDB
    insert_data_to_mongo(df)