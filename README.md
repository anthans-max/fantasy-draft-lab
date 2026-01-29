# Fantasy Draft Lab (Demo)

A lightweight Streamlit demo for fantasy football draft analytics. Uses a small local CSV dataset to power charts and a draft strategy summary.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run ui/app.py
```

## Project Structure

- `ui/app.py`: Streamlit UI with data, charts, and draft advisor tabs
- `data/sample_players.csv`: Sample player dataset used for the demo
- `src/`: Reserved for future analytics/helpers

## Notes

- No API keys are required.
- The UI includes a scoring format toggle (PPR vs Half-PPR) that drives all charts.
- The Draft Advisor tab includes placeholders for a future Gemini-powered explanation.
