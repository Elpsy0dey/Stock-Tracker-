import os
import imaplib
import email
import re
import pandas as pd
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from pathlib import Path

# Load credentials from .env
load_dotenv()
EMAIL = os.getenv("stake_email")
PASSWORD = os.getenv("stake_app_password")

def connect_to_gmail():
    """Connect to Gmail via IMAP"""
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(EMAIL, PASSWORD)
        print("✅ Successfully connected to Gmail")
        return mail
    except Exception as e:
        print(f"❌ Error connecting to Gmail: {str(e)}")
        return None

def fetch_stake_emails(mail, start_date=None, max_emails=1000):
    """Fetch Stake trade confirmation emails with date filtering"""
    try:
        # Search all mail folders (not just inbox)
        mail.select('"[Gmail]/All Mail"')
        
        # Use correct IMAP search syntax: each key is a separate argument
        if start_date:
            formatted_date = start_date.strftime("%d-%b-%Y")
            print(f"🔍 Searching with criteria: FROM notifications@hellostake.com SINCE {formatted_date}")
            status, data = mail.search(None, 'FROM', 'notifications@hellostake.com', 'SINCE', formatted_date)
        else:
            print(f"🔍 Searching with criteria: FROM notifications@hellostake.com")
            status, data = mail.search(None, 'FROM', 'notifications@hellostake.com')
        email_ids = data[0].split()[-max_emails:]  # last N emails

        messages = []
        for eid in email_ids:
            _, msg_data = mail.fetch(eid, "(RFC822)")
            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)

            # Get email subject and date
            subject = msg['subject']
            date = msg['date']
            
            # Ignore cancelled orders
            if subject and 'order cancelled' in subject.lower():
                print(f"⏩ Skipping cancelled order email: {subject} ({date})")
                continue
            print(f"\n📧 Processing email: {subject} ({date})")

            # Get HTML content with encoding handling
            for part in msg.walk():
                if part.get_content_type() == "text/html":
                    payload = part.get_payload(decode=True)
                    try:
                        html = payload.decode("utf-8")
                    except UnicodeDecodeError:
                        try:
                            html = payload.decode("iso-8859-1")
                        except UnicodeDecodeError:
                            # Try other common encodings if needed
                            html = payload.decode("latin-1")
                    messages.append(html)
                    break

        print(f"\n✅ Successfully fetched {len(messages)} Stake emails")
        return messages
    except Exception as e:
        print(f"❌ Error fetching emails: {str(e)}")
        return []

def parse_trade_email(html):
    """Parse trade details from Stake email HTML"""
    try:
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text()

        def extract(pattern, group=1, default=None, cast=str):
            match = re.search(pattern, text)
            if match:
                try:
                    return cast(match.group(group).replace(',', '').replace('US$', '').strip())
                except:
                    return default
            return default

        # Parse Trade Date - More resilient parsing
        trade_date_str = extract(r"on (\d{1,2} \w{3} \d{4})", 1)
        if trade_date_str:
            try:
                # First try parsing with dayfirst=True for more flexibility
                trade_date = pd.to_datetime(trade_date_str, dayfirst=True, errors='coerce')
                if pd.isna(trade_date):
                    # If that fails, try with explicit format
                    trade_date = pd.to_datetime(trade_date_str, format="%d %b %Y", errors='coerce')
            except Exception as e:
                print(f"⚠️ Warning: Could not parse date '{trade_date_str}': {str(e)}")
                trade_date = None
        else:
            trade_date = None

        # Calculate Settlement Date = T+2 business days
        settlement_date = pd.bdate_range(start=trade_date, periods=3)[-1] if trade_date else None

        trade_details = {
            "Trade Date": trade_date,
            "Settlement Date": settlement_date,
            "Symbol": extract(r"Your (.*?) order has been filled", 1),
            "Side": extract(r"LIMIT (\w+)", 1),
            "Trade Identifier": extract(r"Order number\s+([A-Z0-9]+)"),
            "Units": extract(r"Shares\s+(\d+)", cast=int),
            "Avg. Price": extract(r"Effective price\s+US\$(\d+\.\d+)", cast=float),
            "Value": extract(r"Trade value\s+US\$(\d+\.\d+)", cast=float),
            "Fees": extract(r"Brokerage\s+US\$(\d+\.\d+)", cast=float),
            "GST": extract(r"Regulatory fees\s+US\$(\d+\.\d+)", cast=float),
            "Total Value": extract(r"Total value\s+US\$(\d+\.\d+)", cast=float),
            "Currency": "USD",
            "AUD/USD rate": None  # placeholder
        }

        # Print extracted details for verification
        print("\n📊 Extracted Trade Details:")
        for key, value in trade_details.items():
            print(f"{key}: {value}")

        return trade_details
    except Exception as e:
        print(f"❌ Error parsing trade details: {str(e)}")
        return None

def build_trades_dataframe(messages, min_trade_date=None):
    """Build a DataFrame from parsed trade emails with date filtering"""
    try:
        records = [parse_trade_email(html) for html in messages]
        records = [r for r in records if r is not None]  # Filter out None values
        
        if not records:
            print("⚠️ No valid trade records found")
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        
        # Filter by trade date if specified
        if min_trade_date and "Trade Date" in df.columns:
            df = df[df["Trade Date"] >= min_trade_date]
            print(f"📅 Filtered to {len(df)} trades after {min_trade_date}")
        
        # Sort by trade date if available
        if "Trade Date" in df.columns:
            df = df.sort_values("Trade Date", ascending=False)
        
        return df
    except Exception as e:
        print(f"❌ Error building DataFrame: {str(e)}")
        return pd.DataFrame()

def export_trade_data(df, base_filename="stake_trade_records"):
    """Export trade data to CSV in the trade_exports folder, removing old CSVs first"""
    export_dir = Path("trade_exports")
    export_dir.mkdir(exist_ok=True)
    # Remove old CSVs
    for old_csv in export_dir.glob(f"{base_filename}_*.csv"):
        try:
            os.remove(old_csv)
            print(f"🗑️ Removed old CSV: {old_csv}")
        except Exception as e:
            print(f"⚠️ Could not remove {old_csv}: {e}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Export to CSV only
    csv_path = export_dir / f"{base_filename}_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Exported {len(df)} records to CSV: {csv_path}")
    return {'csv_path': str(csv_path)}

def main():
    print("🔍 Stake Trade Records Export")
    print("----------------------------")
    
    # Set date filters
    start_date = datetime(2025, 5, 1)  # May 1st, 2025
    min_trade_date = pd.Timestamp("2025-04-30")  # April 30th, 2025
    
    print(f"📅 Filtering emails from: {start_date.strftime('%d-%b-%Y')}")
    print(f"📅 Filtering trades from: {min_trade_date.strftime('%Y-%m-%d')}")
    
    # 1️⃣ Connect and pull filtered Stake emails
    mail = connect_to_gmail()
    if not mail:
        return
    
    messages = fetch_stake_emails(mail, start_date=start_date, max_emails=1000)
    
    if not messages:
        print("❌ No messages found")
        return
    
    # 2️⃣ Build the DataFrame with trade date filtering
    df = build_trades_dataframe(messages, min_trade_date=min_trade_date)
    
    if df.empty:
        print("❌ No valid trade records found")
        return
    
    # 3️⃣ Export data to multiple formats
    export_paths = export_trade_data(df)
    
    # 4️⃣ Display preview of the data
    print("\n📊 Preview of Trade Records:")
    print(df.head())
    
    # 5️⃣ Display summary statistics
    print("\n📈 Trade Summary:")
    print(f"Total Trades: {len(df)}")
    print(f"Date Range: {df['Trade Date'].min()} to {df['Trade Date'].max()}")
    print(f"Total Value: ${df['Total Value'].sum():,.2f}")
    print(f"Total Fees: ${df['Fees'].sum():,.2f}")
    
    # Close connection
    mail.close()
    mail.logout()
    print("\n✅ Export completed successfully")

if __name__ == "__main__":
    main() 