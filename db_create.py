import sqlite3
import pandas as pd
import os
import streamlit as st
import tempfile

def create_database_and_insert_csv(uploaded_file, db_path="lead_data.db"):
    """
    Modified version that accepts Streamlit UploadedFile object
    """
    # Create a temporary file to store the uploaded content
    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
        # Write uploaded file to temporary file
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    try:
        # Step 1: Read Excel file from temporary path
        df = pd.read_excel(tmp_path)
        print(f"Excel loaded successfully. Shape: {df.shape}")
        print(f"Original Columns: {list(df.columns)}")
        
        # Rest of your existing function code...
        # [Keep all the cleaning and database operations the same]
        
        # Step 2: Clean column names
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('-', '_')
        
        # Step 3: Clean monetary columns
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
        
        # Step 4-7: Database operations (same as your original function)
        conn = sqlite3.connect(db_path)
        
        # Create table
        cursor = conn.cursor()
        cursor.execute("""
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
        """)
        conn.commit()
        
        # Insert data
        df.to_sql('lead_data', conn, if_exists='replace', index=False)
        
        # Verify
        cursor.execute("SELECT COUNT(*) FROM lead_data")
        count = cursor.fetchone()[0]
        print(f"Inserted {count} rows")
        
        conn.close()
        return True
        
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return False
    finally:
        # Clean up the temporary file
        os.unlink(tmp_path)

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
