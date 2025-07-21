from openai import OpenAI
import sqlite3
import os

def init_openai():
    """Initialize OpenAI client with GPT-4o."""
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key and api_key != "your_openai_api_key_here":
        return OpenAI(api_key=api_key)
    return None


def generate_sql_query(user_query):
    """Generate SQL query from natural language using OpenAI GPT-4"""
    
    # Define the prompt template with schema information
    prompt_template = """
    You are a SQL query generator. Generate a SQL query based on the user's natural language request.
    
    Database Schema:
    Table: lead_data
    Columns:
    - PUBLISHER (TEXT): Lead publisher/source
    - TARGET (TEXT): Target information
    - BUYER (TEXT): Buyer information
    - CAMPAIGN (TEXT): Campaign name
    - CALL_ID (TEXT): Unique call identifier
    - RECORDING_ID (TEXT): Recording identifier
    - CALL_DATE (TEXT): Date of call
    - QUOTE (TEXT): Quote information
    - SALE (TEXT): Sale information
    - REVENUE (REAL): Revenue amount
    - PAYOUT (REAL): Payout amount
    - BILLABLE (TEXT): Billable status
    - DURATION (INTEGER): Call duration in seconds
    - IVR (TEXT): IVR status
    - WAIT_IN_SEC (INTEGER): Wait time in seconds
    - LEAD_REVIEW (TEXT): Lead review information
    - CUSTOMER_INTENT (TEXT): Customer intent
    - REACHED_AGENT (TEXT): Whether customer reached agent
    - ISQUALIFIED (TEXT): Whether lead is qualified
    - SCORE (REAL): Lead score
    - CALLER_ID (TEXT): Caller ID
    - DISPOSITION (TEXT): Call disposition
    - CALLBACK_CONVERSION (TEXT): Callback conversion info
    - AD_MENTION (TEXT): Ad mention information
    - AD_MISLED (TEXT): Ad misled information
    - CALL_BACK (TEXT): Callback information
    - IVR2 (TEXT): Secondary IVR information
    - OBJECTION_WITH_NO_REBUTTAL (TEXT): Objection information
    - QUOTE3 (TEXT): Quote 3 information
    - SALE4 (TEXT): Sale 4 information
    - STAGE_1_INTRODUCTION (TEXT): Stage 1 data
    - STAGE_2_ELIGIBILITY (TEXT): Stage 2 data
    - STAGE_3_NEEDS_ANALYSIS (TEXT): Stage 3 data
    - STAGE_4_PLAN_DETAIL (TEXT): Stage 4 data
    - STAGE_5_ENROLLMENT (TEXT): Stage 5 data
    - TALKED_TO_AGENT (TEXT): Whether talked to agent
    - UNPROFESSIONALISM (TEXT): Unprofessionalism flag
    - YOU_CALLED_ME (TEXT): You called me flag
    - YOU_CALLED_ME_PROMPT (TEXT): You called me prompt
    
    User Query: {user_query}
    
    Generate only the SQL query without any explanation or markdown formatting.
    """
    
    # Format the full prompt with the user's input
    full_prompt = prompt_template.format(user_query=user_query)
    
    try:
        # Initialize OpenAI client
        client = init_openai()
        
        # Make the API call using the new syntax
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You generate SQL queries from user input using a fixed schema."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.0,  # deterministic output
        )
        
        # Extract the SQL query
        sql_response = response.choices[0].message.content.strip()
        
        return sql_response
        
    except Exception as e:
        raise Exception(f"OpenAI API Error: {str(e)}")


def run_sql_query(sql_query, db_path="lead_data.db"):
    """Executes SQL query and returns list of dicts (column-based)."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # allows name-based access
        cursor = conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        conn.close()
        
        # Convert to list of dictionaries
        result = [dict(row) for row in rows]
        return result
    except Exception as e:
        raise RuntimeError(f"Failed to run SQL: {e}")
