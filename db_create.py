import sqlite3
import pandas as pd
import os

def create_database_and_insert_csv(csv_file_path, db_path="lead_data.db"):
    """
    Create SQLite database and insert cleaned Excel data
    """
    
    # Step 1: Read Excel file
    try:
        df = pd.read_excel(csv_file_path)
        print(f"CSV loaded successfully. Shape: {df.shape}")
        print(f"Original Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return False

    # Step 2: Clean column names (spaces and dashes → underscores)
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('-', '_')
    print(f"Cleaned Columns: {list(df.columns)}")

    # Step 3: Clean and cast monetary columns
    for col in ['REVENUE', 'PAYOUT']:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r'[\$,]', '', regex=True)
                .str.strip()
                .replace('', '0')
                .astype(float)
            )

    # Step 4: Create database connection
    try:
        conn = sqlite3.connect(db_path)
        print(f"Database created/connected: {db_path}")
    except Exception as e:
        print(f"Error creating database: {e}")
        return False

    # Step 5: Create table with cleaned column names
    create_table_query = """
    CREATE TABLE IF NOT EXISTS lead_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        PUBLISHER TEXT,
        TARGET TEXT,
        BUYER TEXT,
        CAMPAIGN TEXT,
        CALL_ID TEXT,
        RECORDING_ID TEXT,
        CALL_DATE TEXT,
        QUOTE TEXT,
        SALE TEXT,
        REVENUE REAL,
        PAYOUT REAL,
        BILLABLE TEXT,
        DURATION INTEGER,
        IVR TEXT,
        WAIT_IN_SEC INTEGER,
        LEAD_REVIEW TEXT,
        CUSTOMER_INTENT TEXT,
        REACHED_AGENT TEXT,
        ISQUALIFIED TEXT,
        SCORE REAL,
        CALLER_ID TEXT,
        DISPOSITION TEXT,
        CALLBACK_CONVERSION TEXT,
        AD_MENTION TEXT,
        AD_MISLED TEXT,
        CALL_BACK TEXT,
        IVR2 TEXT,
        OBJECTION_WITH_NO_REBUTTAL TEXT,
        QUOTE3 TEXT,
        SALE4 TEXT,
        STAGE_1_INTRODUCTION TEXT,
        STAGE_2_ELIGIBILITY TEXT,
        STAGE_3_NEEDS_ANALYSIS TEXT,
        STAGE_4_PLAN_DETAIL TEXT,
        STAGE_5_ENROLLMENT TEXT,
        TALKED_TO_AGENT TEXT,
        UNPROFESSIONALISM TEXT,
        YOU_CALLED_ME TEXT,
        YOU_CALLED_ME_PROMPT TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    try:
        cursor = conn.cursor()
        cursor.execute(create_table_query)
        conn.commit()
        print("Table created successfully")
    except Exception as e:
        print(f"Error creating table: {e}")
        conn.close()
        return False

    # Step 6: Insert cleaned data
    try:
        df.to_sql('lead_data', conn, if_exists='replace', index=False)
        print(f"Data inserted successfully. Total rows: {len(df)}")

        # Verify insertion
        cursor.execute("SELECT COUNT(*) FROM lead_data")
        count = cursor.fetchone()[0]
        print(f"Verified: {count} rows in database")

    except Exception as e:
        print(f"Error inserting data: {e}")
        conn.close()
        return False

    # Step 7: Close connection
    conn.close()
    print("Database connection closed")
    return True


def query_database(db_path="lead_data.db", query="SELECT * FROM lead_data LIMIT 5"):
    """
    Query the database to verify data
    """
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error querying database: {e}")
        return None


# Example usage
if __name__ == "__main__":
    # Replace with your Excel file path
    csv_file = "sample_data.xlsx"
    
    # Create database and insert data
    success = create_database_and_insert_csv(csv_file)
    
    if success:
        # Query to verify data
        sample_data = query_database()
        if sample_data is not None:
            print("\nSample data from database:")
            print(sample_data.head())
        
        # Show table info
        conn = sqlite3.connect("lead_data.db")
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(lead_data)")
        columns = cursor.fetchall()
        conn.close()

        print("\nTable structure:")
        for col in columns:
            print(f"Column: {col[1]}, Type: {col[2]}")
