import requests

def send_telegram_message(message, token, chat_id):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    # Parse mode to HTML
    payload["parse_mode"] = "HTML"
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Errore invio messaggio Telegram: {e}")
