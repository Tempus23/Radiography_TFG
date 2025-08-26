BASE_URL = 'https://debitum.investments/gtw/loans/api/balances'
LOGIN_URL = 'https://www.mintos.com/webapp/api/auth/login'
VERIFY = 'https://www.mintos.com/webapp/api/marketplace-api/v1/user/verification/sessions'
from requests import get, post

user = "carlos.cbm32@gmail.com"
password = "CODCRAFtroll23."

def fetch_data():
    url = f"{BASE_URL}"
    
    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "en",
        "authorization": "Bearer FsIrc43gSUkV5_HK6VTibCqHwQ0",
        "cookie": "CookieConsent={stamp:'buz6Fc6SRXvPbhJoBRyQCBt0xhZeBxy+3rcyP/HlU2a1q7gx2ZQchA==',necessary:true,preferences:false,statistics:false,marketing:false,method:'explicit',ver:1,utc:1751724180335,region:'es'}; intercom-id-fa5sbayp=2b4cb0cf-2ba7-4b6e-971e-e6fcb2bfa454; intercom-session-fa5sbayp=; intercom-device-id-fa5sbayp=4360bee1-b8ad-4fea-9279-0a9a3eea7fff; cfzs_google-analytics_v4={\"cViC_pageviewCounter\":{\"v\":\"58\"},\"cViC_conversionCounter\":{\"v\":\"77\"}}; cfz_google-analytics_v4={\"cViC_engagementDuration\":{\"v\":\"0\",\"e\":1787764885715},\"cViC_engagementStart\":{\"v\":\"1756228885715\",\"e\":1787764885715},\"cViC_counter\":{\"v\":\"152\",\"e\":1787764885715},\"cViC_session_counter\":{\"v\":\"37\",\"e\":1787764885715},\"cViC_ga4\":{\"v\":\"9ed1d495-fb05-4f31-a4dd-a045d503ac8e\",\"e\":1787764885715},\"cViC__z_ga_audiences\":{\"v\":\"9ed1d495-fb05-4f31-a4dd-a045d503ac8e\",\"e\":1782923930281},\"cViC_let\":{\"v\":\"1756228885715\",\"e\":1787764885715},\"cViC_ga4sid\":{\"v\":\"1916737057\",\"e\":1756230685715}}",
        "kadabra": "1",
        "priority": "u=1, i",
        "referer": "https://debitum.investments/en/overview",
        "sec-ch-ua": "\"Opera GX\";v=\"120\", \"Not-A.Brand\";v=\"8\", \"Chromium\";v=\"135\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36 OPR/120.0.0.0"
    }

    response = get("https://debitum.investments/gtw/account?refresh=true", headers=headers)
    try:
        return response.json()
    except Exception as e:
        print(f"Failed to parse JSON. Status code: {response.status_code}")
        
        print(f"Reponse: {response.text}")

print(fetch_data())