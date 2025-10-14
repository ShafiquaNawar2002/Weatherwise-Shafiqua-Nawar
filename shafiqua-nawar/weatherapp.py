# weatherapp.py
# -----------------------------------------------------------
# UI server for Weather Advisor (uses functions from test.py)
# Run:
#   pip install flask requests
#   python weatherapp.py
#
# Folder structure:
#   your-project/
#     test.py                 # <-- your existing file (as provided)
#     weatherapp.py           # <-- this file
#     templates/
#       index.html            # <-- the UI
# -----------------------------------------------------------

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import BadRequest

# We import directly from the provided test.py
# (Make sure test.py is in the same folder as this file.)
from test import (
    sanitize_location,
    get_weather_data,
    parse_weather_question,
    generate_weather_response,
)

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/temps")
def api_temps():
    """
    Returns JSON for temperature chart:
    {
      "location": "Perth",
      "dates": ["2025-10-14", ...],
      "min": [..], "avg": [..], "max": [..]
    }
    """
    loc_in = request.args.get("location", "", type=str)
    days = request.args.get("days", default=3, type=int)
    location = sanitize_location(loc_in)
    if not location:
        raise BadRequest("Please provide a valid 'location'.")

    wd = get_weather_data(location, forecast_days=days)
    if not wd:
        return jsonify({"error": f"Could not fetch weather for '{location}'."}), 502

    dates, mins, avgs, maxs = [], [], [], []
    for day in wd.get("forecast", []):
        dates.append(day.get("date", ""))
        def f(x):
            try:
                return float(x)
            except Exception:
                return 0.0
        mins.append(f(day.get("mintempC", 0)))
        avgs.append(f(day.get("avgtempC", 0)))
        maxs.append(f(day.get("maxtempC", 0)))

    return jsonify({
        "location": wd.get("location", location.title()),
        "dates": dates, "min": mins, "avg": avgs, "max": maxs
    })


@app.get("/api/rain")
def api_rain():
    """
    Returns JSON for precipitation chart:
    {
      "location": "Perth",
      "dates": ["2025-10-14", ...],
      "chance": [0..100, ...]  # daily max chance of rain
    }
    """
    loc_in = request.args.get("location", "", type=str)
    days = request.args.get("days", default=3, type=int)
    location = sanitize_location(loc_in)
    if not location:
        raise BadRequest("Please provide a valid 'location'.")

    wd = get_weather_data(location, forecast_days=days)
    if not wd:
        return jsonify({"error": f"Could not fetch weather for '{location}'."}), 502

    def max_rain(day):
        max_r = 0
        for h in (day.get("hourly") or []):
            try:
                c = int(h.get("chanceofrain", 0))
                if c > max_r:
                    max_r = c
            except Exception:
                pass
        return max_r

    dates, chances = [], []
    for day in wd.get("forecast", []):
        dates.append(day.get("date", ""))
        chances.append(max_rain(day))

    return jsonify({
        "location": wd.get("location", location.title()),
        "dates": dates, "chance": chances
    })


@app.post("/api/ask")
def api_ask():
    """
    Chat endpoint.
    Body JSON: { "question": "...", "default_location": "Perth" (optional) }
    Returns: { "answer": "..." }
    """
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    default_loc_in = (data.get("default_location") or "").strip()

    if not question:
        raise BadRequest("Missing 'question'.")

    parsed = parse_weather_question(question)
    # If the question didn't include a detectable location,
    # use a provided default (if any).
    loc = parsed.get("location")
    if not loc:
        loc = sanitize_location(default_loc_in)
        parsed["location"] = loc

    if not parsed.get("location"):
        return jsonify({
            "answer": "Please include a location in your question or set a default location."
        })

    days = parsed.get("days", 3)
    wd = get_weather_data(parsed["location"], forecast_days=days)
    if not wd:
        return jsonify({"answer": "Sorry, I couldn't retrieve weather data right now."}), 502

    answer = generate_weather_response(parsed, wd)
    return jsonify({"answer": answer})


@app.get("/healthz")
def healthz():
    return jsonify({"ok": True})


if __name__ == "__main__":
    # Windows-friendly dev server
    app.run(host="127.0.0.1", port=5000, debug=True)
