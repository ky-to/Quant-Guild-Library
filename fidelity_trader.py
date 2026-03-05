"""
Fidelity Trade Automation via Selenium

Fidelity has NO official trading API. This uses browser automation
to interact with Fidelity's web interface.

Requirements:
    pip install selenium webdriver-manager

WARNINGS:
    - Fidelity may change their website at any time, breaking selectors
    - Browser automation is fragile — test with small orders first
    - Use at your own risk with real money
    - 2FA / security prompts may require manual intervention
    - Consider using a broker with a real API (Interactive Brokers, Alpaca, etc.)
"""

import time
import logging
from dataclasses import dataclass
from enum import Enum
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── Configuration ───────────────────────────────────────────────────────────

class OrderAction(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class TradeOrder:
    symbol: str
    action: OrderAction
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None


# ─── Fidelity URLs ──────────────────────────────────────────────────────────

FIDELITY_LOGIN_URL = "https://digital.fidelity.com/prgw/digital/login/full-page"
FIDELITY_TRADE_URL = "https://digital.fidelity.com/ftgw/digital/trade-equity/index/orderEntry"
FIDELITY_POSITIONS_URL = "https://digital.fidelity.com/ftgw/digital/portfolio/positions"


# ─── Browser Setup ──────────────────────────────────────────────────────────

def create_driver(headless: bool = False) -> webdriver.Chrome:
    """Create a Chrome WebDriver instance."""
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    # Reduce bot detection
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.implicitly_wait(10)
    return driver


# ─── Login ──────────────────────────────────────────────────────────────────

def login(driver: webdriver.Chrome, username: str, password: str, timeout: int = 30):
    """
    Log into Fidelity. May require manual 2FA completion.
    """
    logger.info("Navigating to Fidelity login page...")
    driver.get(FIDELITY_LOGIN_URL)
    wait = WebDriverWait(driver, timeout)

    # Enter username
    username_field = wait.until(EC.presence_of_element_located((By.ID, "dom-username-input")))
    username_field.clear()
    username_field.send_keys(username)

    # Enter password
    password_field = wait.until(EC.presence_of_element_located((By.ID, "dom-pswd-input")))
    password_field.clear()
    password_field.send_keys(password)

    # Click login
    login_button = wait.until(EC.element_to_be_clickable((By.ID, "dom-login-button")))
    login_button.click()

    logger.info("Login submitted. Waiting for 2FA or dashboard...")

    # Wait for 2FA or successful redirect
    # If 2FA is required, the user must complete it manually
    _wait_for_login_complete(driver, timeout=60)
    logger.info("Login successful.")


def _wait_for_login_complete(driver: webdriver.Chrome, timeout: int = 60):
    """Wait until we're past login (handles 2FA pause)."""
    wait = WebDriverWait(driver, timeout)
    # Wait until URL changes away from login page
    wait.until(lambda d: "login" not in d.current_url.lower() or "summary" in d.current_url.lower())
    time.sleep(2)  # Let the page settle


# ─── Place Trade ────────────────────────────────────────────────────────────

def place_trade(driver: webdriver.Chrome, order: TradeOrder, account_index: int = 0, timeout: int = 20):
    """
    Place a trade on Fidelity's equity trade page.

    Args:
        driver: Logged-in Selenium WebDriver
        order: TradeOrder with trade details
        account_index: Which account to trade in (0 = first/default)
        timeout: Max seconds to wait for elements
    """
    logger.info(f"Placing order: {order.action.value} {order.quantity} {order.symbol} ({order.order_type.value})")
    driver.get(FIDELITY_TRADE_URL)
    wait = WebDriverWait(driver, timeout)
    time.sleep(3)  # Let trade page load fully

    # ── Select Account (if multiple) ──
    # Fidelity shows an account dropdown if you have multiple accounts
    # Skip if only one account
    try:
        account_dropdown = driver.find_elements(By.CSS_SELECTOR, "[data-testid='account-selector']")
        if account_dropdown:
            account_dropdown[0].click()
            time.sleep(1)
            accounts = driver.find_elements(By.CSS_SELECTOR, "[data-testid='account-option']")
            if account_index < len(accounts):
                accounts[account_index].click()
                time.sleep(1)
    except Exception:
        logger.debug("Single account or account selector not found, continuing...")

    # ── Enter Symbol ──
    symbol_input = wait.until(EC.presence_of_element_located(
        (By.CSS_SELECTOR, "input[aria-label='Symbol']")
    ))
    symbol_input.clear()
    symbol_input.send_keys(order.symbol)
    symbol_input.send_keys(Keys.TAB)
    time.sleep(2)  # Wait for symbol lookup

    # ── Select Action (Buy/Sell) ──
    action_dropdown = wait.until(EC.element_to_be_clickable(
        (By.CSS_SELECTOR, "[data-testid='action-selector'], select[name='action']")
    ))
    action_dropdown.click()
    time.sleep(0.5)

    action_option = wait.until(EC.element_to_be_clickable(
        (By.XPATH, f"//option[contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'{order.action.value}')]")
    ))
    action_option.click()
    time.sleep(0.5)

    # ── Enter Quantity ──
    qty_input = wait.until(EC.presence_of_element_located(
        (By.CSS_SELECTOR, "input[aria-label='Quantity'], input[name='quantity']")
    ))
    qty_input.clear()
    qty_input.send_keys(str(order.quantity))

    # ── Select Order Type ──
    order_type_map = {
        OrderType.MARKET: "market",
        OrderType.LIMIT: "limit",
        OrderType.STOP: "stop",
        OrderType.STOP_LIMIT: "stop limit",
    }

    order_type_dropdown = wait.until(EC.element_to_be_clickable(
        (By.CSS_SELECTOR, "[data-testid='order-type-selector'], select[name='orderType']")
    ))
    order_type_dropdown.click()
    time.sleep(0.5)

    type_text = order_type_map[order.order_type]
    order_type_option = wait.until(EC.element_to_be_clickable(
        (By.XPATH, f"//option[contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'{type_text}')]")
    ))
    order_type_option.click()
    time.sleep(0.5)

    # ── Enter Limit Price ──
    if order.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and order.limit_price is not None:
        limit_input = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, "input[aria-label='Limit price'], input[name='limitPrice']")
        ))
        limit_input.clear()
        limit_input.send_keys(str(order.limit_price))

    # ── Enter Stop Price ──
    if order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and order.stop_price is not None:
        stop_input = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, "input[aria-label='Stop price'], input[name='stopPrice']")
        ))
        stop_input.clear()
        stop_input.send_keys(str(order.stop_price))

    # ── Preview Order ──
    preview_button = wait.until(EC.element_to_be_clickable(
        (By.XPATH, "//button[contains(text(),'Preview') or contains(text(),'preview')]")
    ))
    preview_button.click()
    logger.info("Order preview opened. Review the details...")
    time.sleep(3)

    # ── Confirm / Place Order ──
    place_button = wait.until(EC.element_to_be_clickable(
        (By.XPATH, "//button[contains(text(),'Place order') or contains(text(),'Place Order') or contains(text(),'Submit')]")
    ))
    place_button.click()
    logger.info(f"Order placed: {order.action.value} {order.quantity} {order.symbol}")
    time.sleep(3)


# ─── Get Positions ──────────────────────────────────────────────────────────

def get_positions(driver: webdriver.Chrome, timeout: int = 20) -> list[dict]:
    """
    Scrape current positions from the Fidelity portfolio page.
    Returns a list of dicts with symbol, quantity, market_value, etc.
    """
    logger.info("Fetching positions...")
    driver.get(FIDELITY_POSITIONS_URL)
    wait = WebDriverWait(driver, timeout)
    time.sleep(5)  # Let positions page load

    positions = []
    try:
        # Fidelity renders positions in a table
        rows = wait.until(EC.presence_of_all_elements_located(
            (By.CSS_SELECTOR, "tr[data-testid='position-row'], .posweb-cell-symbol-name")
        ))

        for row in rows:
            try:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) >= 4:
                    positions.append({
                        "symbol": cells[0].text.strip(),
                        "quantity": cells[1].text.strip(),
                        "last_price": cells[2].text.strip(),
                        "market_value": cells[3].text.strip(),
                    })
            except Exception:
                continue

    except Exception as e:
        logger.warning(f"Could not parse positions: {e}")

    logger.info(f"Found {len(positions)} positions.")
    return positions


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from getpass import getpass

    # Credentials — use environment variables or prompt
    USERNAME = os.environ.get("FIDELITY_USERNAME") or input("Fidelity Username: ")
    PASSWORD = os.environ.get("FIDELITY_PASSWORD") or getpass("Fidelity Password: ")

    # Create browser (headless=False so you can handle 2FA)
    driver = create_driver(headless=False)

    try:
        # 1. Log in
        login(driver, USERNAME, PASSWORD)

        # 2. Check positions
        positions = get_positions(driver)
        for pos in positions:
            print(pos)

        # 3. Place a trade (example: buy 1 share of AAPL at market)
        order = TradeOrder(
            symbol="AAPL",
            action=OrderAction.BUY,
            quantity=1,
            order_type=OrderType.MARKET,
        )
        place_trade(driver, order)

        # Example: limit order
        # limit_order = TradeOrder(
        #     symbol="MSFT",
        #     action=OrderAction.BUY,
        #     quantity=5,
        #     order_type=OrderType.LIMIT,
        #     limit_price=400.00,
        # )
        # place_trade(driver, limit_order)

    finally:
        input("Press Enter to close the browser...")
        driver.quit()
