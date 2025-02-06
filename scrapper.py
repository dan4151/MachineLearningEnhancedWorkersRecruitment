from time import sleep

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time
import pandas as pd
from tqdm import tqdm
import re
import csv
import random

# Chrome options for reducing detection
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36")
options.add_argument("--disable-blink-features=AutomationControlled")
# Uncomment the next line to add proxy support
# options.add_argument("--proxy-server=http://your_proxy_ip:port")

driver = webdriver.Chrome(options=options)

# Read the university data
df = pd.read_csv("universities.csv")

# Output file
output_file = "university_data.csv"

#how many lines there are currently in output file
with open(output_file, mode="r", newline="", encoding="utf-8") as file:
    reader = csv.reader(file)
    n = len(list(reader))


for university in tqdm(df["title"][n-1:]):
    driver.get("https://www.google.com/")
    time.sleep(random.uniform(1, 3))  # Random delay before interacting

    # Find the search box and simulate human typing
    search_box = driver.find_element(By.NAME, "q")
    for char in university:
        search_box.send_keys(char)
        time.sleep(random.uniform(0.1, 0.3))  # Simulate typing delay
    search_box.send_keys(Keys.RETURN)
    time.sleep(random.uniform(5, 10))  # Allow page to load

    avg_cost, grad_rate, acc_rate = None, None, None

    # Extract Avg Cost
    try:
        avg_cost_elem = driver.find_element(By.XPATH,
                                            "//*[contains(text(), 'Avg cost after aid')]/following-sibling::*")
        avg_cost = avg_cost_elem.text
        if avg_cost == "––":
            raise Exception("No data")
        num = float(re.sub(r'[^\d.]', '', avg_cost))
        if "K" in avg_cost:
            num *= 1000
        avg_cost = num
    except Exception:
        avg_cost = None

    # Extract Graduation Rate
    try:
        grad_rate_elem = driver.find_element(By.XPATH, "//*[contains(text(), 'Graduation rate')]/following-sibling::*")
        grad_rate = grad_rate_elem.text
        if grad_rate == "––":
            raise Exception("No data")
        grad_rate = float(grad_rate[:-1]) / 100  # Convert percentage to decimal
    except Exception:
        grad_rate = None

    # Extract Acceptance Rate
    try:
        acc_rate_elem = driver.find_element(By.XPATH, "//*[contains(text(), 'Acceptance rate')]/following-sibling::*")
        acc_rate = acc_rate_elem.text
        if acc_rate == "––":
            raise Exception("No data")
        acc_rate = float(acc_rate[:-1]) / 100  # Convert percentage to decimal
    except Exception:
        acc_rate = None

    # Save the data row to the CSV file
    with open(output_file, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([university, avg_cost, grad_rate, acc_rate])

print("Scraping complete")
driver.quit()
