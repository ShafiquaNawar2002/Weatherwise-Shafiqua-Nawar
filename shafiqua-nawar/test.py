# Weather Advisor - Fast + Humanized Answers (with location sanitiser)
# -------------------------------------------------------------------
# Run:
#   pip install requests matplotlib
#   python weather_advisor_fast_human.py
#
# Optional (Ollama for parsing; otherwise a fast rule-based parser is used):
#   ollama serve
#   ollama pull llama3.1
#   # set WEATHER_ADVISOR_DISABLE_OLLAMA=1 to disable Ollama

import os
import json
import re
import requests
import matplotlib.pyplot as plt

WTTR_TIMEOUT_SECS = 6
OLLAMA_TIMEOUT_SECS = 4
MAX_FORECAST_DAYS = 5

# --- NEW: sanitize location to prevent "Perth Tomorrow" 404s ---
_TIME_TOKENS = {
    "today", "tomorrow", "tonight", "weekend", "morning", "afternoon", "evening",
    "next", "day", "days", "week", "weeks", "month", "months",
    "monday","tuesday","wednesday","thursday","friday","saturday","sunday",
    "this"  # e.g., "this weekend"
}
_PREP_TOKENS = {"in", "at", "on", "for"}

def sanitize_location(raw: str) -> str:
    """
    Remove time words / prepositions / stray numbers from a user-entered 'location'.
    Keeps letters, spaces, hyphens, and apostrophes (for names like O'Connor).
    """
    if not isinstance(raw, str):
        return ""
    # Keep alpha, space, hyphen, apostrophe, comma
    cleaned = re.sub(r"[^A-Za-z\s\-\',]", " ", raw.strip())
    parts = [p for p in re.split(r"[,\s]+", cleaned) if p]
    keep = []
    for p in parts:
        low = p.lower()
        if low in _TIME_TOKENS:     # drop time words
            continue
        if low in _PREP_TOKENS:     # drop prepositions often pasted in
            continue
        if low.isdigit():           # drop numbers from phrases like "next 3 days"
            continue
        keep.append(p)
    loc = " ".join(keep).strip(" ,")
    return loc

# ------------------------------
# 1) Data retrieval (wttr.in)
# ------------------------------
def get_weather_data(location, forecast_days=5):
    """
    Retrieve weather data for a specified location.

    Args:
        location (str): City or location name
        forecast_days (int): Number of days to forecast (1-5)

    Returns:
        dict or None
    """
    # NEW: sanitize the location (fixes "Perth Tomorrow")
    location = sanitize_location(location)
    if not location:
        print("Please provide a valid location (e.g., 'Perth').")
        return None

    days = max(1, min(MAX_FORECAST_DAYS, int(forecast_days or 3)))

    url = f"https://wttr.in/{location}?format=j1"
    try:
        r = requests.get(url, timeout=WTTR_TIMEOUT_SECS)
        r.raise_for_status()
        data = r.json()

        current = (data.get("current_condition") or [{}])[0]
        forecast_raw = data.get("weather") or []
        forecast = forecast_raw[:days]  # wttr.in usually returns up to 3 days

        return {"location": location.title(), "current": current, "forecast": forecast}
    except Exception as e:
        print(f"[Network] Could not get weather data for '{location}': {e}")
        return None

# -------------------------------------------
# 2) Visualisation: Temperature (min/avg/max)
# -------------------------------------------
def create_temperature_visualisation(weather_data, output_type='display'):
    if not weather_data or "forecast" not in weather_data:
        print("No weather data supplied.")
        return None

    dates, mins, avgs, maxs = [], [], [], []
    for day in weather_data["forecast"]:
        dates.append(day.get("date", ""))
        def f(x):
            try: return float(x)
            except: return 0.0
        mins.append(f(day.get("mintempC", 0)))
        avgs.append(f(day.get("avgtempC", 0)))
        maxs.append(f(day.get("maxtempC", 0)))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(dates, mins, marker='o', label='Min °C')
    ax.plot(dates, avgs, marker='o', label='Avg °C')
    ax.plot(dates, maxs, marker='o', label='Max °C')
    ax.set_title(f"Temperature Trend - {weather_data.get('location', '')}")
    ax.set_xlabel("Date"); ax.set_ylabel("Temperature (°C)")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()

    if output_type == 'figure':
        return fig
    plt.show()
    return None

# ---------------------------------------------------
# 3) Visualisation: Precipitation chance (daily max)
# ---------------------------------------------------
def create_precipitation_visualisation(weather_data, output_type='display'):
    if not weather_data or "forecast" not in weather_data:
        print("No weather data supplied.")
        return None

    dates, rain_chance = [], []
    for day in weather_data["forecast"]:
        dates.append(day.get("date", ""))
        max_chance = 0
        for h in (day.get("hourly") or []):
            try:
                c = int(h.get("chanceofrain", 0))
                if c > max_chance: max_chance = c
            except: pass
        rain_chance.append(max_chance)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(dates, rain_chance, marker='o', label='Chance of Rain (%)')
    ax.set_title(f"Precipitation Chances - {weather_data.get('location', '')}")
    ax.set_xlabel("Date"); ax.set_ylabel("Chance of Rain (%)")
    ax.set_ylim(0, 100); ax.grid(True, linestyle='--', alpha=0.4); ax.legend()

    if output_type == 'figure':
        return fig
    plt.show()
    return None

# ------------------------------
# 4) Natural Language Interface
# ------------------------------
def _ollama_disabled():
    return os.environ.get("WEATHER_ADVISOR_DISABLE_OLLAMA", "").strip() == "1"

def _call_ollama(prompt, model=None, temperature=0.0, json_only=True):
    """Return text from Ollama or None on any issue (fast timeout)."""
    if _ollama_disabled():
        return None
    try:
        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        url = f"{host}/api/generate"
        model = model or os.environ.get("WEATHER_ADVISOR_OLLAMA_MODEL", "llama3.1")
        payload = {"model": model, "prompt": prompt,
                   "options": {"temperature": temperature}, "stream": False}
        res = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT_SECS)
        res.raise_for_status()
        text = res.json().get("response", "").strip()
        if json_only:
            s, e = text.find("{"), text.rfind("}")
            if s != -1 and e != -1: text = text[s:e+1]
        return text
    except Exception:
        return None  # instant fallback

def parse_weather_question(question):
    """
    Returns dict: {location, days, when, attribute}
    when ∈ {'today','tomorrow','next_n_days'}
    attribute ∈ {'temperature','precipitation','rain','wind','humidity','summary'}
    """
    question = (question or "").strip()
    if not question:
        return {"location": None, "days": 3, "when": "today", "attribute": "summary"}

    # Try LLM parse first (quick timeout)
    system = (
        "You are a weather question parser. Output JSON with keys: "
        "location (string|null), days (1..5), when ('today'|'tomorrow'|'next_n_days'), "
        "attribute ('temperature'|'rain'|'precipitation'|'wind'|'humidity'|'summary'). "
        "If 'this weekend' -> next_n_days + days=3. Default: location=null, days=3, when='today', attribute='summary'."
    )
    prompt = f"{system}\nUser question: {question}\nJSON:"
    llm_text = _call_ollama(prompt, json_only=True)

    if llm_text:
        try:
            p = json.loads(llm_text)
            loc = sanitize_location(p.get("location") or "")
            loc = loc or None
            days = max(1, min(MAX_FORECAST_DAYS, int(p.get("days", 3))))
            when = p.get("when", "today")
            if when not in ("today","tomorrow","next_n_days"): when = "today"
            attr = p.get("attribute","summary")
            if attr not in ("temperature","rain","precipitation","wind","humidity","summary"):
                attr = "summary"
            return {"location": loc, "days": days, "when": when, "attribute": attr}
        except Exception:
            pass  # fall through

    # Rule-based fallback
    q = question.lower()
    # capture text after "in|at" but stop before time words if present
    loc = None
    m = re.search(r"\b(?:in|at)\s+([A-Za-z][A-Za-z\s\-']{1,60})", q)
    if m:
        loc = sanitize_location(m.group(1))
        if not loc:
            loc = None

    days, when = 3, "today"
    if "tomorrow" in q: when, days = "tomorrow", 1
    m = re.search(r"\bnext\s+(\d)\s+day", q)
    if m:
        when = "next_n_days"
        days = max(1, min(MAX_FORECAST_DAYS, int(m.group(1))))
    if "weekend" in q: when, days = "next_n_days", 3

    if any(w in q for w in ["rain", "umbrella", "precip"]):
        attribute = "precipitation"
    elif any(w in q for w in ["windy", "wind"]):
        attribute = "wind"
    elif any(w in q for w in ["humid", "humidity"]):
        attribute = "humidity"
    elif any(w in q for w in ["hot", "cold", "warm", "temp", "temperature"]):
        attribute = "temperature"
    else:
        attribute = "summary"

    return {"location": loc, "days": days, "when": when, "attribute": attribute}

def _pick_day_slice(weather_data, when, days):
    days_list = weather_data.get("forecast", [])
    if not days_list: return []
    if when == "tomorrow" and len(days_list) >= 2: return days_list[1:2]
    if when == "next_n_days": return days_list[:days]
    return days_list[:1]  # today

def _max_rain_chance(day):
    max_r = 0
    for h in (day.get("hourly") or []):
        try:
            c = int(h.get("chanceofrain", 0))
            if c > max_r: max_r = c
        except: pass
    return max_r

def _wind_max_kmph(day):
    max_w = 0
    for h in (day.get("hourly") or []):
        try:
            w = int(h.get("windspeedKmph", 0))
            if w > max_w: max_w = w
        except: pass
    return max_w

def _avg_humidity(day):
    total = count = 0
    for h in (day.get("hourly") or []):
        try:
            total += int(h.get("humidity", 0)); count += 1
        except: pass
    return int(round(total / count)) if count else 0

def _yes_maybe_no_from_rain(chance):
    if chance >= 60:   return "Yes—bring an umbrella."
    if chance >= 30:   return "Maybe—pack one just in case."
    return "Probably not—rain chance is low."

def _wind_label(kmph):
    if kmph >= 60: return "very windy"
    if kmph >= 40: return "windy"
    if kmph >= 25: return "a bit breezy"
    return "light winds"

# NEW: compact daily description line
def _day_brief(day):
    date = day.get("date", "Unknown date")
    avgc = day.get("avgtempC", "?")
    minc = day.get("mintempC", "?")
    maxc = day.get("maxtempC", "?")
    chance = _max_rain_chance(day)
    # try to grab a midday-ish description
    desc = None
    for h in (day.get("hourly") or []):
        if h.get("time") in ("1200","900","1500"):
            wdesc = h.get("weatherDesc")
            if isinstance(wdesc, list) and wdesc:
                desc = wdesc[0].get("value")
            break
    if not desc:
        # fallback: any description
        for h in (day.get("hourly") or []):
            wdesc = h.get("weatherDesc")
            if isinstance(wdesc, list) and wdesc:
                desc = wdesc[0].get("value"); break
    desc = desc or "—"
    return f"{date}: ~{avgc}°C (min {minc}°C / max {maxc}°C), rain up to {chance}%, {desc}"

# UPDATED: Answer first, then forecast block
def generate_weather_response(parsed_question, weather_data):
    """
    HUMANIZED answer first, then a concise Forecast section.
    """
    if not weather_data:
        return "Sorry, I couldn't retrieve weather data right now."

    loc = weather_data.get("location", "your location")
    when = parsed_question.get("when", "today")
    days = parsed_question.get("days", 3)
    attribute = parsed_question.get("attribute", "summary")

    selected_days = _pick_day_slice(weather_data, when, days)
    if not selected_days:
        return f"Sorry, I couldn't find a forecast for {loc}."

    d0 = selected_days[0]

    # --- Humanised lead line ---
    if attribute in ("precipitation", "rain"):
        chance = _max_rain_chance(d0)
        lead = f"{_yes_maybe_no_from_rain(chance)} ({chance}% chance of rain {when.replace('_',' ')} in {loc})."
    elif attribute == "temperature":
        avgc = d0.get("avgtempC", "?"); minc = d0.get("mintempC", "?"); maxc = d0.get("maxtempC", "?")
        lead = f"Expect about {avgc}°C in {loc} {when.replace('_',' ')} (min {minc}°C / max {maxc}°C)."
    elif attribute == "wind":
        w = _wind_max_kmph(d0)
        lead = f"It'll be {_wind_label(w)} in {loc} {when.replace('_',' ')} (gusts up to ~{w} km/h)."
    elif attribute == "humidity":
        h = _avg_humidity(d0)
        lead = f"Humidity in {loc} {when.replace('_',' ')} will average around {h}%."
    else:
        avgc = d0.get("avgtempC", "?"); minc = d0.get("mintempC", "?"); maxc = d0.get("maxtempC", "?")
        chance = _max_rain_chance(d0)
        lead = (f"In {loc} {when.replace('_',' ')}, expect about {avgc}°C "
                f"(min {minc}°C / max {maxc}°C) with up to {chance}% chance of rain.")

    # --- Forecast block ---
    header_when = {"today":"Today","tomorrow":"Tomorrow"}.get(when, when.replace("_"," ").title())
    lines = [lead, "", f"Forecast for {loc} — {header_when}:" if len(selected_days)==1
             else f"Forecast for {loc} — next {len(selected_days)} days:"]
    for day in selected_days:
        lines.append("• " + _day_brief(day))

    return "\n".join(lines)

# ------------------------------
# 5) Helpers + Simple UI
# ------------------------------
def print_current_and_forecast(wd):
    if not wd:
        print("No data."); return
    loc = wd.get("location", "Unknown")
    cur = wd.get("current", {}) or {}
    desc = "N/A"
    if isinstance(cur.get("weatherDesc"), list) and cur["weatherDesc"]:
        desc = cur["weatherDesc"][0].get("value", "N/A")
    temp_c = cur.get("temp_C", "N/A")

    print(f"\nCurrent weather in {loc}: {temp_c}°C, {desc}")
    print("\nForecast:")
    for d in wd.get("forecast", []):
        print("  " + _day_brief(d))

def run_menu():
    print("\n====== Weather Advisor ======")
    while True:
        print("\nMenu:")
        print("  1) Current + Forecast summary")
        print("  2) Show Temperature Trend (chart)")
        print("  3) Show Precipitation Chances (chart)")
        print("  4) Ask a Question (natural language)")
        print("  5) Quit")
        choice = input("Choose an option (1-5): ").strip()

        if choice == "5":
            print("Goodbye!"); break
        if choice not in ("1","2","3","4"):
            print("Please choose 1-5."); continue

        if choice in ("1","2","3"):
            loc_in = input("Enter city/location (e.g., Perth): ").strip()
            loc = sanitize_location(loc_in)
            days_in = input("How many forecast days (1-5)? ").strip()
            try: days = int(days_in)
            except: days = 3
            wd = get_weather_data(loc, forecast_days=days)
            if not wd: continue

            if choice == "1":
                print_current_and_forecast(wd)
            elif choice == "2":
                print("(Close the chart window to return here.)")
                create_temperature_visualisation(wd, output_type='display')
            elif choice == "3":
                print("(Close the chart window to return here.)")
                create_precipitation_visualisation(wd, output_type='display')
        else:
            q = input("\nAsk about the weather (e.g., 'Do I need an umbrella tomorrow in Perth?')\n> ").strip()
            parsed = parse_weather_question(q)
            loc = parsed.get("location")
            if not loc:
                # If user types "tomorrow" without a location, ask.
                loc_input = input("Which location? ").strip()
                loc = sanitize_location(loc_input) or "Perth"
                parsed["location"] = loc
            days = parsed.get("days", 3)

            wd = get_weather_data(parsed["location"], forecast_days=days)
            if not wd:
                print("Could not fetch weather. Try again.")
                continue
            answer = generate_weather_response(parsed, wd)
            print("\n" + answer + "\n")

if __name__ == "__main__":
    run_menu()
