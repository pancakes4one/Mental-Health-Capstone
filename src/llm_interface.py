import os
import json
import joblib
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)

MODEL = "meta-llama/Llama-3.3-70B-Instruct"

# The fields our trained model needs, with the exact category values it was trained on.
# The LLM uses this list to know what to look for and how to map casual language
# (e.g. "I work remote" -> remote_work: "Yes") to the exact values the model expects.
REQUIRED_FIELDS = {
    "age": "a number, e.g. 29",
    "gender": "male, female, or other",
    "self_employed": "Yes or No",
    "family_history": "Yes or No — family history of mental illness",
    "work_interfere": "Never, Rarely, Sometimes, Often, or Not applicable — how much mental health interferes with work",
    "no_employees": "1-5, 6-25, 26-100, 100-500, 500-1000, or More than 1000 — company size",
    "remote_work": "Yes or No",
    "tech_company": "Yes or No",
    "benefits": "Yes, No, or Unsure — employer provides mental health benefits",
    "care_options": "Yes, No, or Unsure — knows options for mental health care through employer",
    "wellness_program": "Yes, No, or Unsure",
    "seek_help": "Yes, No, or Unsure — employer provides resources to seek help",
    "anonymity": "Yes, No, or Unsure — anonymity protected if seeking help",
    "leave": "Very easy, Somewhat easy, Unsure, Somewhat difficult, or Very difficult — ease of taking medical leave",
    "mental_health_consequence": "Yes, No, or Unsure — negative consequences for discussing mental health with employer",
    "phys_health_consequence": "Yes, No, or Unsure",
    "coworkers": "Yes, No, or Some of them — willing to discuss with coworkers",
    "supervisor": "Yes, No, or Some of them — willing to discuss with supervisor",
    "mental_health_interview": "Yes, No, or Unsure — would bring up in a job interview",
    "phys_health_interview": "Yes, No, or Unsure",
    "mental_vs_physical": "Yes, No, or Unsure — employer takes mental health as seriously as physical health",
    "obs_consequence": "Yes or No — observed negative consequences for coworkers with mental health conditions",
    "is_us": "Yes or No — based in the United States",
}


def parse_user_input(user_message):
    """
    Sends the user's message to the LLM and asks it to extract whichever
    REQUIRED_FIELDS it can find. Returns a dict with two keys:
      - "found": {field_name: value, ...} for fields it could extract
      - "missing": [field_name, ...] for fields it could not find
    """
    field_list = "\n".join(f"- {name}: {desc}" for name, desc in REQUIRED_FIELDS.items())

    system_prompt = f"""You extract structured survey answers from a user's message.

Here are the fields to look for, and the exact allowed values for each:
{field_list}

Read the user's message and extract any of these fields you can confidently determine.
Map casual language to the exact allowed values (e.g. "I work remote" -> remote_work: "Yes").
Do not guess a value if it isn't stated or clearly implied.

Respond with ONLY a JSON object in this exact format, no other text:
{{"found": {{"field_name": "value", ...}}, "missing": ["field_name", ...]}}

"missing" should list every field from the list above that you could NOT determine.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0
    )

    raw_text = response.choices[0].message.content.strip()

    # Strip markdown code fences if the model wraps the JSON in them
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        raw_text = raw_text.replace("json\n", "", 1)

    return json.loads(raw_text)


def load_artifacts():
    """Loads the fitted age scaler and the trained feature column order."""
    age_scaler = joblib.load('models/age_scaler.joblib')
    with open('models/feature_columns.json') as f:
        feature_columns = json.load(f)
    return age_scaler, feature_columns


# Mappings used during training -- must match preprocess.py exactly
WORK_INTERFERE_MAP = {'Not applicable': 0, 'Never': 1, 'Rarely': 2, 'Sometimes': 3, 'Often': 4}
LEAVE_MAP = {'Very easy': 0, 'Somewhat easy': 1, 'Unsure': 2, 'Somewhat difficult': 3, 'Very difficult': 4}
NO_EMPLOYEES_MAP = {'1-5': 0, '6-25': 1, '26-100': 2, '100-500': 3, '500-1000': 4, 'More than 1000': 5}
BINARY_COLS = ['self_employed', 'family_history', 'remote_work', 'tech_company', 'obs_consequence', 'is_us']
NOMINAL_FIELDS = ['gender', 'benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity',
                   'mental_vs_physical', 'mental_health_consequence', 'phys_health_consequence',
                   'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview']


def encode_features(found, feature_columns, age_scaler):
    """
    Takes the "found" dict from parse_user_input (all REQUIRED_FIELDS must be present)
    and builds a single-row DataFrame matching the exact columns/order the model was
    trained on -- same binary mapping, ordinal mapping, one-hot columns, and age scaling.
    """
    row = {col: 0 for col in feature_columns}

    for col in BINARY_COLS:
        if col in row:
            row[col] = 1 if found.get(col) == 'Yes' else 0

    if 'work_interfere' in row:
        row['work_interfere'] = WORK_INTERFERE_MAP.get(found.get('work_interfere'), 0)

    if 'leave' in row:
        row['leave'] = LEAVE_MAP.get(found.get('leave'), 0)

    if 'no_employees' in row:
        row['no_employees'] = NO_EMPLOYEES_MAP.get(found.get('no_employees'), 0)

    for field in NOMINAL_FIELDS:
        value = found.get(field)
        col_name = f"{field}_{value}"
        if col_name in row:
            row[col_name] = 1

    age_value = float(found['age'])
    scaled_age = age_scaler.transform([[age_value]])[0][0]
    if 'age' in row:
        row['age'] = scaled_age

    return pd.DataFrame([row], columns=feature_columns)


if __name__ == "__main__":
    # Full example with every required field, to test encoding independently of parsing
    example_found = {
        "age": "29", "gender": "female", "self_employed": "No", "family_history": "Yes",
        "work_interfere": "Sometimes", "no_employees": "26-100", "remote_work": "Yes",
        "tech_company": "Yes", "benefits": "No", "care_options": "Unsure",
        "wellness_program": "No", "seek_help": "Unsure", "anonymity": "Unsure",
        "leave": "Somewhat easy", "mental_health_consequence": "Unsure",
        "phys_health_consequence": "No", "coworkers": "Some of them", "supervisor": "Yes",
        "mental_health_interview": "No", "phys_health_interview": "No",
        "mental_vs_physical": "Unsure", "obs_consequence": "No", "is_us": "Yes"
    }

    age_scaler, feature_columns = load_artifacts()
    encoded_row = encode_features(example_found, feature_columns, age_scaler)
    print(encoded_row.T)