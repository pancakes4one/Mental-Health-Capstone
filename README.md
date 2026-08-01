# Mental Health Treatment Predictor

An end-to-end ML application that predicts whether someone is likely to seek
mental health treatment, based on workplace and demographic factors — with a
natural-language chat interface built on top of the trained model.

## Demo
[Watch the demo video](https://drive.google.com/file/d/1Pns3XEdv6KiZ59CW8_18MP21q6_2xD5_/view?usp=drive_link)

## What it does

The app combines two pieces:

1. **A trained classification model** predicting `treatment` (whether someone
   has sought mental health treatment) from the [OSMI Mental Health in Tech
   Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
   (2014, 1259 rows) — workplace factors like remote work, company size,
   benefits, and family history.
2. **An LLM-powered chat interface** (Streamlit + Nebius Token Factory) that
   asks a few natural-language questions, parses the answers, runs them
   through the trained model, and explains the prediction in plain English.

Who it's for: anyone curious how workplace and demographic factors relate 
to mental health treatment-seeking, or reviewing this as a demonstration of an end-to-end ML +
LLM pipeline (this was built as a course capstone project).

## Setup

**1. Clone the repo and create a virtual environment:**
```bash
git clone https://github.com/pancakes4one/Mental-Health-Capstone.git
cd Mental-Health-Capstone
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Get the dataset:**
The raw data isn't committed to this repo (see `.gitignore`). Download
`survey.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
and place it at `data/raw/survey.csv`.

**3. Set up your API key:**
Copy `.env.example` to `.env` and add your Nebius Token Factory API key:
```bash
cp .env.example .env
```
Then edit `.env`:
```
NEBIUS_API_KEY=your-key-here
```
Get a key at [tokenfactory.nebius.com](https://tokenfactory.nebius.com) —
billing setup is required to use the API, but new accounts get $1 in trial
credit, which comfortably covers extensive testing of this project (a few
cents at most for typical use).

**4. Train the models:**
```bash
python src/train_models.py
```
This cleans the data, trains 5 model configurations, logs everything to
MLflow, and saves the artifacts (`data/processed/`, `models/age_scaler.joblib`,
`models/feature_columns.json`) that the chat interface needs.

## Usage

**Run the chat interface:**
```bash
streamlit run src/app.py
```
Opens in your browser. Answer the questions it asks (age, gender, family
history, etc.) and it'll return a prediction with an explanation.

**View experiment tracking:**
```bash
mlflow ui
```
Then open `http://localhost:5000` (or `--port 5001` if 5000 is taken) to see
all logged runs, metrics, and parameters.

**Run the test suite:**
```bash
pytest tests/ -v
```

## Architecture

```
data/raw/survey.csv
        ↓
src/preprocess.py    — cleaning, encoding, age scaling
        ↓
src/train_models.py  — trains 5 model configs, logs each to MLflow
        ↓
src/evaluate.py      — mlflow.search_runs() picks the best run by F1 score
        ↓
src/llm_interface.py — parses user messages into model features, validates
                        input, runs the best model, generates a plain-English
                        response via the LLM
        ↓
src/app.py            — Streamlit chat UI wrapping llm_interface.py
```

The LLM (Nebius Token Factory, `Llama-3.3-70B-Instruct`) is used two ways:
extracting structured feature values from free-text user messages, and
turning the raw model prediction into a conversational explanation.

## Results

5 model configurations were trained and logged to MLflow:

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Random Forest (default) | 0.8008 | 0.7698 | 0.8629 | 0.8137 | 0.8774 |
| **Random Forest (deeper)** | **0.8130** | **0.7786** | **0.8790** | **0.8258** | 0.8757 |
| Gradient Boosting (default) | 0.7724 | 0.7464 | 0.8306 | 0.7863 | 0.8660 |
| Gradient Boosting (slow LR) | 0.7886 | 0.7535 | 0.8629 | 0.8045 | 0.8775 |
| Neural Network | 0.8089 | 0.7984 | 0.8306 | 0.8142 | 0.8734 |

**Best model: Random Forest (deeper)** — highest F1 score (0.8258) and
accuracy (0.8130), selected programmatically via `mlflow.search_runs()`
ordered by F1. Random Forest also has the practical advantage of not
requiring feature scaling to perform well, unlike the Neural Network.

## Known limitations & design decisions

- **Only 7 of the model's ~23 input features are actively asked about** in
  the chat interface (age, gender, family history, work interference,
  benefits, remote work, US location). The remaining, more subjective
  fields (e.g. "does your employer take mental health as seriously as
  physical health") default to a neutral value (`Unsure`) unless the user
  happens to mention them. This trades some prediction precision for a much
  shorter, more usable conversation — a deliberate choice for a demo
  interface, not an oversight.
- **Age validation allows 13–100**, but the model's training data only
  covers ages 18–100. Predictions for ages under 18 are flagged in the
  response as an extrapolation beyond the model's training range.
- **`country` was collapsed to a binary `is_us` flag** rather than one-hot
  encoding every country, since most countries in the dataset had very few
  respondents.
- **Model availability on Nebius Token Factory changes over time** (the
  original small parsing model used during development was later removed
  from the catalog). The interface now uses one consistent model
  (`Llama-3.3-70B-Instruct`) for both parsing and response generation to
  reduce moving parts.

## Reflection

**What I learned:** The modeling itself was fairly straightforward. Cleaning the data, selecting features, training Random Forest models, and comparing metrics went smoothly. The bigger challenge was making sure the app handled new user input the same way the training data was processed, with the same encoding and scaling.

I also learned that simpler can be better. While building the LLM interface, I reduced the number of questions from 22 to 12, then to 7, and switched from asking multiple questions at once to one at a time. Each change made the application easier to use while keeping its purpose.

I came to understand that building an AI application is about much more than training a model. Managing the development environment, keeping workflows reproducible, and designing a simple user experience are just as important as the machine learning.

**What was challenging:** Challenges arose from simple changes such as renaming my project folder which broke my virtual environment, causing Python version conflicts, package mismatches, and MLflow compatibility issues. Fixing those problems took as much time as building the model.

It was also a challenge to keep the feature encoding used at prediction time perfectly consistent with what the model was actually trained on. I encountered a bug where I was fitting the scaler on the whole dataset instead of only the training split which is classic data leakage.

Working with a live third-party API also meant handling instability: model names changing, an unexpected reasoning-model output format, and version mismatches between a local MLflow database and the installed MLflow package.

I iterated a lot on what to ask the user, eventually cutting it to the most predictive 7 fields.

**What I'd improve with more time:** I would compare a reduced "Feature Set B". I would try combining overlapping features, for example, like coworkers and supervisor, which both measure comfort discussing mental health with people at work, into a single feature. I'd compare that reduced feature set against the full one to see how a simpler model performs.

I'd also focus more on the US-specific data, since about 60% of respondents were US-based. state was in the original dataset but I dropped it early on due to missing values for non-US respondents. I would want to add it back in for a US-only analysis and see whether treatment-seeking actually differs meaningfully from state to state.

I'd also want to look more closely at tech_company specifically. The dataset explicitly asks whether someone works at a tech company, which made me expect it to be one of the more meaningful splits in the data — I didn't dig into whether treatment-seeking actually looks different between tech and non-tech respondents, and I think that's worth exploring.

I'd also want to spend more time testing faster, cheaper models for the parsing step. I ran into a few models being deprecated or removed from Nebius's catalog mid-project, which pushed me toward one larger, reliable model for simplicity.

Extend the chat interface to optionally ask about the remaining fields for users who want a more precise prediction rather than the fast 7-question default.