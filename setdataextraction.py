import requests
from bs4 import BeautifulSoup
import pandas as pd

# Step 1: Define URL for the Stock Exchange of Thailand (SET)
BASE_URL = "https://www.set.or.th/en/market/index"  # Replace with the actual URL for SET data

# Step 2: Fetch Data
def fetch_set_data():
    try:
        # Send a GET request to the website
        response = requests.get(BASE_URL)
        response.raise_for_status()  # Raise an error if the request was unsuccessful
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract relevant data (e.g., stock prices, market indices, etc.)
        # This is an example and will depend on the structure of the SET webpage
        table = soup.find('table', {'class': 'table-class'})  # Replace 'table-class' with the actual class of the data table
        rows = table.find_all('tr')

        # Extract headers
        headers = [header.text.strip() for header in rows[0].find_all('th')]

        # Extract rows of data
        data = []
        for row in rows[1:]:
            cols = row.find_all('td')
            data.append([col.text.strip() for col in cols])

        # Convert to DataFrame
        df = pd.DataFrame(data, columns=headers)
        return df
    except Exception as e:
        print("Error fetching data from SET:", e)
        return None

# Step 3: Save Data to CSV
def save_to_csv(df, filename="set_data.csv"):
    if df is not None:
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
    else:
        print("No data to save.")

# Main Function
def main():
    # Fetch data from SET
    df = fetch_set_data()

    # Display the fetched data
    if df is not None:
        print(df.head())

    # Save the data to a CSV file
    save_to_csv(df)

if __name__ == "__main__":
    main()
