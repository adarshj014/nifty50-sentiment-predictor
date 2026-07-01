from gdeltdoc import GdeltDoc, Filters
from datetime import datetime
import time
import random

# removed domain filter — too restrictive for Indian news in GDELT
# we filter by country instead which is more reliable

NIFTY50_COMPANIES = {
    "RELIANCE.NS"   : ("Reliance Industries",        "Reliance RIL stock"),
    "TCS.NS"        : ("Tata Consultancy Services",   "TCS India stock"),
    "HDFCBANK.NS"   : ("HDFC Bank India",             "HDFC Bank results"),
    "INFY.NS"       : ("Infosys India",               "Infosys results"),
    "ICICIBANK.NS"  : ("ICICI Bank India",            "ICICI Bank stock"),
    "HINDUNILVR.NS" : ("Hindustan Unilever India",    "HUL results India"),
    "ITC.NS"        : ("ITC Limited India",           "ITC stock NSE"),
    "SBIN.NS"       : ("State Bank of India",         "SBI bank results"),
    "BHARTIARTL.NS" : ("Bharti Airtel India",         "Airtel India stock"),
    "KOTAKBANK.NS"  : ("Kotak Mahindra Bank",         "Kotak Bank results"),
    "LT.NS"         : ("Larsen and Toubro India",     "L&T India results"),
    "AXISBANK.NS"   : ("Axis Bank India",             "Axis Bank results"),
    "ASIANPAINT.NS" : ("Asian Paints India",          "Asian Paints results"),
    "MARUTI.NS"     : ("Maruti Suzuki India",         "Maruti stock results"),
    "SUNPHARMA.NS"  : ("Sun Pharmaceutical India",    "Sun Pharma results"),
    "TITAN.NS"      : ("Titan Company India",         "Titan stock NSE"),
    "BAJFINANCE.NS" : ("Bajaj Finance India",         "Bajaj Finance results"),
    "NESTLEIND.NS"  : ("Nestle India",                "Nestle India results"),
    "WIPRO.NS"      : ("Wipro India",                 "Wipro results NSE"),
    "ULTRACEMCO.NS" : ("UltraTech Cement India",      "UltraTech results"),
    "POWERGRID.NS"  : ("Power Grid Corporation India","Power Grid results"),
    "NTPC.NS"       : ("NTPC India",                  "NTPC results NSE"),
    "TECHM.NS"      : ("Tech Mahindra India",         "Tech Mahindra results"),
    "HCLTECH.NS"    : ("HCL Technologies India",      "HCL Tech results"),
    "BAJAJFINSV.NS" : ("Bajaj Finserv India",         "Bajaj Finserv results"),
    "ONGC.NS"       : ("ONGC India",                  "ONGC results NSE"),
    "JSWSTEEL.NS"   : ("JSW Steel India",             "JSW Steel results"),
    "COALINDIA.NS"  : ("Coal India",                  "Coal India NSE"),
    "GRASIM.NS"     : ("Grasim Industries India",     "Grasim results NSE"),
    "DIVISLAB.NS"   : ("Divi Laboratories India",     "Divi Labs results"),
    "DRREDDY.NS"    : ("Dr Reddys Laboratories",      "Dr Reddy results"),
    "EICHERMOT.NS"  : ("Eicher Motors India",         "Royal Enfield stock"),
    "CIPLA.NS"      : ("Cipla India",                 "Cipla results NSE"),
    "BPCL.NS"       : ("Bharat Petroleum India",      "BPCL results NSE"),
    "HEROMOTOCO.NS" : ("Hero MotoCorp India",         "Hero Moto results"),
    "BRITANNIA.NS"  : ("Britannia Industries India",  "Britannia results"),
    "APOLLOHOSP.NS" : ("Apollo Hospitals India",      "Apollo Hospital stock"),
    "TATACONSUM.NS" : ("Tata Consumer Products",      "Tata Consumer results"),
    "HINDALCO.NS"   : ("Hindalco Industries India",   "Hindalco results"),
    "TATASTEEL.NS"  : ("Tata Steel India",            "Tata Steel results"),
    "UPL.NS"        : ("UPL Limited India",           "UPL stock NSE"),
    "SBILIFE.NS"    : ("SBI Life Insurance India",    "SBI Life results"),
    "BAJAJ-AUTO.NS" : ("Bajaj Auto India",            "Bajaj Auto results"),
    "HDFCLIFE.NS"   : ("HDFC Life Insurance India",   "HDFC Life results"),
    "INDUSINDBK.NS" : ("IndusInd Bank India",         "IndusInd results"),
    "M&M.NS"        : ("Mahindra and Mahindra India", "M&M stock NSE"),
    "ADANIPORTS.NS" : ("Adani Ports India",           "Adani Ports results"),
    "TMPV.NS"       : ("Tata Motors India",           "Tata Motors results"),
    "ADANIENT.NS"   : ("Adani Enterprises India",     "Adani stock NSE"),
    "SHREECEM.NS"   : ("Shree Cement India",          "Shree Cement results"),
}


def query_gdelt(gd, keyword, start_date, end_date):
    """
    Queries GDELT without domain filter.
    Uses country=India filter instead — much better coverage.
    Returns DataFrame or empty list.
    """
    try:
        f = Filters(
            keyword     = keyword,
            start_date  = start_date,
            end_date    = end_date,
            num_records = 75,
            country     = "India"
        )
        result = gd.article_search(f)
        return result if result is not None else []
    except:
        return []


def parse_date(raw):
    try:
        return datetime.strptime(str(raw), "%Y%m%dT%H%M%SZ")
    except:
        return None


def fetch_gdelt_news(ticker, company_name, alt_name,
                     start_date="2023-01-01", end_date="2026-06-30"):
    """
    Fetches headlines monthly for even distribution.
    No domain filter — country=India instead.
    2 keywords per month, cap 100 per month.
    """
    gd        = GdeltDoc()
    results   = []
    seen_urls = set()

    current = datetime.strptime(start_date, "%Y-%m-%d")
    end     = datetime.strptime(end_date,   "%Y-%m-%d")

    while current < end:
        # next month calculation
        if current.month == 12:
            next_m = datetime(current.year + 1, 1, 1)
        else:
            next_m = datetime(current.year, current.month + 1, 1)
        next_m = min(next_m, end)

        s = current.strftime("%Y-%m-%d")
        e = next_m.strftime("%Y-%m-%d")

        month_articles = []

        # query 1 — primary keyword
        rows1 = query_gdelt(gd, company_name, s, e)
        time.sleep(random.uniform(0.3, 0.5))

        # query 2 — secondary keyword
        rows2 = query_gdelt(gd, alt_name, s, e)
        time.sleep(random.uniform(0.3, 0.5))

        for rows in [rows1, rows2]:
            if not hasattr(rows, "iterrows"):
                continue
            for _, row in rows.iterrows():
                if len(month_articles) >= 100:
                    break

                headline = str(row.get("title", "")).strip()
                if not headline or headline == "nan":
                    continue

                url = str(row.get("url", "")).strip()
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                month_articles.append({
                    "ticker"      : ticker,
                    "content"     : headline,
                    "source"      : str(row.get("domain", "")),
                    "published_at": parse_date(row.get("seendate", "")),
                    "url"         : url
                })

        results.extend(month_articles)
        print(f"    {s[:7]}  | {len(month_articles)} articles "
              f"| total: {len(results)}")

        current = next_m

    return results


def fetch_live_news(company_name):
    import feedparser
    url  = (f"https://news.google.com/rss/search?"
            f"q={company_name}+stock+NSE+India"
            f"&hl=en-IN&gl=IN&ceid=IN:en")
    feed = feedparser.parse(url)
    return [{
        "content"     : e.title,
        "published_at": e.published,
        "url"         : e.link,
        "source"      : "Google News"
    } for e in feed.entries[:10]]