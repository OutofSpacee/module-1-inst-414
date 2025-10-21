"""
Scraper to get data from the 40kstats website

Runs in terminal as: python .\warhammer-scraper.py

Outputs four csv files containing:
    win rates and points by faction 
    win rates by detachment
    win rates by faction mission and deployments
    win rate by faction opponent

"""
# imports
from selenium import webdriver # selenium for interactive websites
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
import pandas as pd # for data handeling
import time # for pausing the scraper

# driver for each win-rate-page
print("Opening Chrome drivers")
win_rate_and_points_faction = webdriver.Chrome()
win_rate_faction_mission = webdriver.Chrome()
win_rate_detachment = webdriver.Chrome()
win_rate_opponent = webdriver.Chrome()

# minimizing each open browser
win_rate_and_points_faction.minimize_window()
win_rate_faction_mission.minimize_window()
win_rate_detachment.minimize_window()
win_rate_opponent.minimize_window()

# opening 40k stats sites to data sources
win_rate_and_points_faction.get("https://40kstats.goonhammer.com/#GbF")
win_rate_faction_mission.get("https://40kstats.goonhammer.com/#vpbfm")
win_rate_detachment.get("https://40kstats.goonhammer.com/#subfaction")
win_rate_opponent.get("https://40kstats.goonhammer.com/#FvF")

# setting empty lists to collect data
win_rate_and_points_faction_table = []
win_rate_faction_mission_table = []
win_rate_detachment_table = []
win_rate_opponent_table = []

def scrape_win_by_factions():
    """
    Scrape the simple faction table on the overview page.
    Returns a pandas.DataFrame for all rows.
    """
    global win_rate_and_points_faction_table
    driver = win_rate_and_points_faction
    win_rate_and_points_faction_table = []
    wait = WebDriverWait(driver, 15)
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table thead th")))
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr")))
    except Exception as e:
        print("Timed out waiting for faction table:", e)
        return pd.DataFrame(win_rate_and_points_faction_table)
    header_elements = driver.find_elements(By.CSS_SELECTOR, "table thead th")
    headers = [h.text.strip() for h in header_elements if h.text.strip()]
    if not headers:
        headers = ["Faction", "Games", "VP", "Opp VP", "Win %", "Wins", "Losses", "Draws", "Real Win %"]
    rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
    for row in rows:
        cells = row.find_elements(By.TAG_NAME, "td")
        if not cells:
            continue
        entry = {}
        for i, cell in enumerate(cells):
            text = " ".join(cell.text.split())
            key = headers[i] if i < len(headers) else f"col{i}"
            entry[key] = text
        win_rate_and_points_faction_table.append(entry)
    print(f"Scraped {len(win_rate_and_points_faction_table)} faction rows")
    return pd.DataFrame(win_rate_and_points_faction_table)

def scrape_win_by_faction_mission_all_factions():
    # no-arg wrapper: use internal defaults
    driver = win_rate_faction_mission
    results = win_rate_faction_mission_table
    time.sleep(10)

    # Minimal implementation:
    # - Assume a <select> exists for factions and we can iterate its <option>s.
    # - For each faction: select it, scrape current page table rows, then click a simple "Next" link
    #   (selector 'a.next' or 'a[aria-label="Next"]') repeatedly until it is not enabled.
    def _scrape_current_table_rows(faction_name: str):
        header_elements = driver.find_elements(By.CSS_SELECTOR, "table thead th")
        headers = [h.text.strip() for h in header_elements if h.text.strip()]
        rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
        page_entries = []
        for row in rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            if not cells:
                continue
            entry = {}
            for i, cell in enumerate(cells):
                key = headers[i] if i < len(headers) else f"col{i}"
                entry[key] = " ".join(cell.text.split())
            entry["Faction"] = faction_name
            page_entries.append(entry)
        return page_entries

    # Find a select dropdown and iterate its options
    # Find the correct <select> for the faction dropdown using a few simple heuristics.
    selects = driver.find_elements(By.TAG_NAME, "select")
    select_elem = None

    # Heuristic 1: pick the select whose options include a known faction name (e.g. "Dark Angels").
    for s in selects:
        try:
            opts = [o.text.strip() for o in s.find_elements(By.TAG_NAME, "option") if o.text.strip()]
            lowered = [o.lower() for o in opts]
            if any('dark angel' in o or 'dark angels' in o or 'angels' in o for o in lowered):
                select_elem = s
                break
        except Exception:
            continue

    # Heuristic 2: find a header that mentions "Win Rate by Faction" and take the first select after it.
    if not select_elem:
        try:
            header = driver.find_element(By.XPATH, "//*[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'win rate by faction')]")
            select_elem = header.find_element(By.XPATH, ".//following::select[1]")
        except Exception:
            select_elem = None

    # Fallback: use the last select on the page (often the right-side control)
    if not select_elem and selects:
        select_elem = selects[-1]

    if not select_elem:
        print("No <select> found for factions; minimal scraper requires a <select> element.")
        return pd.DataFrame(results)

    sel = Select(select_elem)
    options = [o for o in sel.options if o.text.strip()]

    for i, opt in enumerate(options):
        faction = opt.text.strip()
        print(f"Selecting faction: {faction}")
        try:
            sel.select_by_index(i)
        except Exception:
            try:
                sel.select_by_visible_text(faction)
            except Exception as e:
                print("Failed to select faction", faction, e)
                continue

        # small pause for page update
        time.sleep(0.6)

        # scrape first page and then paginate via a simple next link
        while True:
            try:
                results.extend(_scrape_current_table_rows(faction))
            except Exception as e:
                print("Error scraping rows:", e)
                break

            # try to find a 'next' link
            next_btn = None
            for selc in ("a.next", "a[aria-label='Next']"):
                elems = driver.find_elements(By.CSS_SELECTOR, selc)
                if elems:
                    next_btn = elems[0]
                    break

            if not next_btn:
                # no next found -> end
                break

            # if next button appears disabled, break
            try:
                cls = (next_btn.get_attribute("class") or "").lower()
                if "disabled" in cls or not next_btn.is_enabled():
                    break
                next_btn.click()
                time.sleep(0.4)
            except Exception as e:
                print("Could not click next button:", e)
                break

    # create DataFrame
    df = pd.DataFrame(results)
    return df

def scrape_win_by_detachments_all_pages() -> pd.DataFrame:
    """
    Scrape the "Win Rate by Detachments" table (uses the global `win_rate_detachment` driver).
    The page defaults to "All Factions", so this function only needs to paginate the table
    by clicking the Next control until no more pages are available.

    Returns a pandas.DataFrame with all rows found.
    """
    driver = win_rate_detachment
    wait = WebDriverWait(driver, 15)
    results = []

    # wait for table to be present
    try:
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table thead th")))
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr")))
    except Exception as e:
        print("Timed out waiting for detachment table:", e)
        return pd.DataFrame(results)

    def _scrape_table():
        headers = [h.text.strip() for h in driver.find_elements(By.CSS_SELECTOR, "table thead th") if h.text.strip()]
        rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
        page = []
        for r in rows:
            cells = r.find_elements(By.TAG_NAME, "td")
            if not cells:
                continue
            entry = {}
            for i, c in enumerate(cells):
                key = headers[i] if i < len(headers) else f"col{i}"
                entry[key] = " ".join(c.text.split())
            page.append(entry)
        return page

    # iterate pages
    while True:
        try:
            page_entries = _scrape_table()
            results.extend(page_entries)
        except Exception as e:
            print("Error scraping detachment rows:", e)
            break

        # find next link
        next_btn = None
        for selc in ("a.next", "a[aria-label='Next']"):
            elems = driver.find_elements(By.CSS_SELECTOR, selc)
            if elems:
                next_btn = elems[0]
                break

        if not next_btn:
            # try xpath text
            elems = driver.find_elements(By.XPATH, "//a[contains(.,'Next') or contains(.,'›')]")
            if elems:
                next_btn = elems[0]

        if not next_btn:
            break

        try:
            cls = (next_btn.get_attribute("class") or "").lower()
            if "disabled" in cls or not next_btn.is_enabled():
                break
            next_btn.click()
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr")))
            time.sleep(0.4)
        except Exception as e:
            print("Failed to click next on detachment page:", e)
            break

    df = pd.DataFrame(results)
    return df

# works as well
def scrape_win_by_faction_opponent_all_factions() -> pd.DataFrame:

    driver = win_rate_opponent
    wait = WebDriverWait(driver, 15)
    results = []

     # wait for table to be present
    try:
         wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table thead th")))
         wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr")))
    except Exception as e:
         print("Timed out waiting for opponent table:", e)
         return pd.DataFrame(results)

    def _scrape_rows(faction_name: str):
        headers = [h.text.strip() for h in driver.find_elements(By.CSS_SELECTOR, "table thead th") if h.text.strip()]
        rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
        entries = []
        for r in rows:
            cells = r.find_elements(By.TAG_NAME, "td")
            if not cells:
                continue
            entry = {}
            for i, c in enumerate(cells):
                key = headers[i] if i < len(headers) else f"col{i}"
                entry[key] = " ".join(c.text.split())
            entry["Faction"] = faction_name
            entries.append(entry)
        return entries

    # find select using heuristics (same as faction/mission scraper)
    selects = driver.find_elements(By.TAG_NAME, "select")
    select_elem = None
    for s in selects:
        try:
            opts = [o.text.strip() for o in s.find_elements(By.TAG_NAME, "option") if o.text.strip()]
            lowered = [o.lower() for o in opts]
            if any('dark angel' in o or 'dark angels' in o or 'angels' in o for o in lowered):
                select_elem = s
                break
        except Exception:
            continue
    if not select_elem:
        try:
            header = driver.find_element(By.XPATH, "//*[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'win rate by faction')]")
            select_elem = header.find_element(By.XPATH, ".//following::select[1]")
        except Exception:
            select_elem = None
    if not select_elem and selects:
        select_elem = selects[-1]
    if not select_elem:
        print("No <select> found for opponent faction dropdown; aborting.")
        return pd.DataFrame(results)
    sel = Select(select_elem)
    options = [o for o in sel.options if o.text.strip()]
    for idx, opt in enumerate(options):
        faction = opt.text.strip()
        print(f"Selecting faction for opponent table: {faction}")
        try:
            sel.select_by_index(idx)
        except Exception:
            try:
                sel.select_by_visible_text(faction)
            except Exception as e:
                print("Failed to select faction", faction, e)
                continue
        time.sleep(0.6)
        while True:
            try:
                results.extend(_scrape_rows(faction))
            except Exception as e:
                print("Error scraping opponent rows:", e)
                break
            # find next control
            next_btn = None
            for selc in ("a.next", "a[aria-label='Next']"):
                elems = driver.find_elements(By.CSS_SELECTOR, selc)
                if elems:
                    next_btn = elems[0]
                    break
            if not next_btn:
                elems = driver.find_elements(By.XPATH, "//a[contains(.,'Next') or contains(.,'›')]")
                if elems:
                    next_btn = elems[0]
            if not next_btn:
                break
            try:
                cls = (next_btn.get_attribute("class") or "").lower()
                if "disabled" in cls or not next_btn.is_enabled():
                    break
                next_btn.click()
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr")))
                time.sleep(0.4)
            except Exception as e:
                print("Failed to click next on opponent page:", e)
                break
    df = pd.DataFrame(results)
    return df

# --- Batch runner for all scrapers ---
def run_all_scrapers_and_save_csvs():
    print("Scraping win rate and points by faction...")
    df1 = scrape_win_by_factions()
    df1.to_csv("collected_data/win_rate_and_points_by_faction.csv", index=False)
    print("Saved win_rate_and_points_by_faction.csv")

    print("Scraping win rate by faction and mission...")
    df2 = scrape_win_by_faction_mission_all_factions()
    df2.to_csv("collected_data/win_rate_by_faction_and_mission.csv", index=False)
    print("Saved win_rate_by_faction_and_mission.csv")

    print("Scraping win rate by detachment...")
    df3 = scrape_win_by_detachments_all_pages()
    df3.to_csv("collected_data/win_rate_by_detachment.csv", index=False)
    print("Saved win_rate_by_detachment.csv")

    print("Scraping win rate by faction vs opponent...")
    df4 = scrape_win_by_faction_opponent_all_factions()
    df4.to_csv("collected_data/win_rate_by_faction_opponent.csv", index=False)
    print("Saved win_rate_by_faction_opponent.csv")

    print("All scraping complete.")

# --- Run all scrapers if called as a script ---
if __name__ == "__main__":
    run_all_scrapers_and_save_csvs()