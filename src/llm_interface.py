import os
import json
import joblib
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY") or "not-set"
)

MODEL = "meta-llama/Llama-3.3-70B-Instruct"  # confirmed reliable, used for everything

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

# Fields we actively ask the user about, batched a few at a time.
# The rest (less critical or opinion-heavy fields) get a reasonable
# neutral default below, unless the user happens to mention them.
CORE_FIELDS = [
    "age", "gender", "family_history", "work_interfere",
    "benefits", "remote_work", "is_us"
]

# Human-readable labels for the questions we actually ask (CORE_FIELDS only)
READABLE_LABELS = {
    "age": "Age",
    "gender": "Gender",
    "family_history": "Family history of mental illness",
    "work_interfere": "How much mental health interferes with work",
    "benefits": "Employer provides mental health benefits",
    "remote_work": "Work remotely",
    "is_us": "Located in the United States",
}

# The exact values each field is allowed to take. Used to reject invalid/nonsense
# answers instead of silently accepting them (e.g. age of 788, or a typo).
ALLOWED_VALUES = {
    "age": None,  # validated separately as a numeric range, see validate_and_clean()
    "gender": ["male", "female", "other"],
    "self_employed": ["Yes", "No"],
    "family_history": ["Yes", "No"],
    "work_interfere": ["Never", "Rarely", "Sometimes", "Often", "Not applicable"],
    "no_employees": ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"],
    "remote_work": ["Yes", "No"],
    "tech_company": ["Yes", "No"],
    "benefits": ["Yes", "No", "Unsure"],
    "care_options": ["Yes", "No", "Unsure"],
    "wellness_program": ["Yes", "No", "Unsure"],
    "seek_help": ["Yes", "No", "Unsure"],
    "anonymity": ["Yes", "No", "Unsure"],
    "leave": ["Very easy", "Somewhat easy", "Unsure", "Somewhat difficult", "Very difficult"],
    "mental_health_consequence": ["Yes", "No", "Unsure"],
    "phys_health_consequence": ["Yes", "No", "Unsure"],
    "coworkers": ["Yes", "No", "Some of them"],
    "supervisor": ["Yes", "No", "Some of them"],
    "mental_health_interview": ["Yes", "No", "Unsure"],
    "phys_health_interview": ["Yes", "No", "Unsure"],
    "mental_vs_physical": ["Yes", "No", "Unsure"],
    "obs_consequence": ["Yes", "No"],
    "is_us": ["Yes", "No"],
}

DEFAULT_VALUES = {
    "self_employed": "No",
    "no_employees": "26-100",
    "tech_company": "Yes",
    "care_options": "Unsure",
    "leave": "Unsure",
    "wellness_program": "Unsure",
    "seek_help": "Unsure",
    "anonymity": "Unsure",
    "mental_health_consequence": "Unsure",
    "phys_health_consequence": "Unsure",
    "coworkers": "Some of them",
    "supervisor": "Some of them",
    "mental_health_interview": "Unsure",
    "phys_health_interview": "Unsure",
    "mental_vs_physical": "Unsure",
    "obs_consequence": "No",
}

QUESTIONS_PER_BATCH = 1

# Natural-language phrasing for each core question, asked one at a time
QUESTIONS = {
    "age": "What's your age?",
    "gender": "What's your gender? (male/female/other)",
    "family_history": "Do you have a family history of mental illness? (Yes/No)",
    "work_interfere": "How much does your mental health interfere with work? (Never/Rarely/Sometimes/Often/Not applicable)",
    "benefits": "Does your employer provide mental health benefits? (Yes/No/Unsure)",
    "remote_work": "Do you work remotely? (Yes/No)",
    "is_us": "Are you located in the United States? (Yes/No)",
}


def parse_user_input(user_message, pending_fields=None):
    """
    Sends the user's message to the LLM and asks it to extract whichever
    REQUIRED_FIELDS it can find. If pending_fields is given (the fields we
    just asked about), the LLM uses that context to interpret short answers
    like "yes" correctly, instead of guessing blind.
    Returns a dict with two keys:
      - "found": {field_name: value, ...} for fields it could extract
      - "missing": [field_name, ...] for fields it could not find
    """
    field_list = "\n".join(f"- {name}: {desc}" for name, desc in REQUIRED_FIELDS.items())

    pending_note = ""
    if pending_fields:
        pending_desc = "\n".join(f"- {name}: {REQUIRED_FIELDS[name]}" for name in pending_fields)
        pending_note = f"""
IMPORTANT: The user was just asked specifically about these fields, in this order:
{pending_desc}
If their message is a short or direct answer (e.g. "yes", "no", "sometimes", or a
list of short answers), interpret it as answering these pending fields, in order.
"""

    system_prompt = f"""You extract structured survey answers from a user's message.

Here are the fields to look for, and the exact allowed values for each:
{field_list}
{pending_note}
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

    # Some models (reasoning models like Qwen3) prepend a <think>...</think>
    # block before the actual answer -- strip that out if present.
    if "</think>" in raw_text:
        raw_text = raw_text.split("</think>")[-1].strip()

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
    scaled_age = age_scaler.transform(pd.DataFrame([[age_value]], columns=['age']))[0][0]
    if 'age' in row:
        row['age'] = scaled_age

    return pd.DataFrame([row], columns=feature_columns)


def load_best_model():
    """Loads the best-performing trained model (by F1 score) from MLflow."""
    from evaluate import get_best_run
    import mlflow.sklearn

    best_run = get_best_run()
    run_id = best_run['run_id']
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    return model, best_run['tags.mlflow.runName']


def predict(encoded_row, model):
    """Runs the model on the encoded row. Returns (prediction, probability)."""
    prediction = model.predict(encoded_row)[0]
    probability = model.predict_proba(encoded_row)[0][1] if hasattr(model, 'predict_proba') else None
    return prediction, probability


def generate_response(found, prediction, probability):
    """Uses the LLM to turn the raw prediction into a clear, conversational explanation."""
    label = "likely to seek mental health treatment" if prediction == 1 else "less likely to seek mental health treatment"
    prob_text = f"{probability * 100:.1f}%" if probability is not None else "unknown"

    age_val = float(found.get("age", 0))
    extrapolation_note = ""
    if age_val < 18:
        extrapolation_note = (
            "\nNote: the underlying model was trained only on survey respondents aged 18-100. "
            "This person's age falls outside that range, so mention that this prediction is an "
            "extrapolation beyond the model's training data and should be treated with extra caution."
        )

    system_prompt = f"""You are explaining a prediction from a machine learning model trained on a workplace mental health survey.

The model predicts this person is: {label}
Model confidence: {prob_text}

Context the user provided: {json.dumps(found)}
{extrapolation_note}
Write a short, clear, conversational response (3-5 sentences) that:
- States the prediction and confidence level in plain language
- Briefly mentions 1-2 factors from their answers that likely influenced it
- Includes a caveat that this is a statistical estimate from survey data, not a clinical diagnosis, and gently encourages talking to a mental health professional if they have real concerns

Do not use markdown formatting. Plain conversational text only.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system_prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


def ask_for_missing(missing_fields, rejected=None):
    """Asks the next single missing field as a natural question, explaining first if their last answer was rejected."""
    next_field = missing_fields[0]
    question = QUESTIONS.get(next_field, f"Can you tell me: {READABLE_LABELS.get(next_field, next_field)}?")

    if rejected and next_field in rejected:
        return f"{rejected[next_field]} {question}"
    return question


# Common abbreviations/shorthand the LLM sometimes doesn't map consistently.
# Checked before falling back to case-insensitive matching against ALLOWED_VALUES.
SYNONYMS = {
    "gender": {"f": "female", "m": "male", "fem": "female"},
    "work_interfere": {"na": "Not applicable", "n/a": "Not applicable"},
}


def normalize_value(field, value):
    """Tries to map a value to its exact allowed form: synonym lookup, then case-insensitive match."""
    if field not in ALLOWED_VALUES or ALLOWED_VALUES[field] is None:
        return value

    value_lower = str(value).strip().lower()

    if field in SYNONYMS and value_lower in SYNONYMS[field]:
        return SYNONYMS[field][value_lower]

    for allowed in ALLOWED_VALUES[field]:
        if allowed.lower() == value_lower:
            return allowed

    return value  # no match found, leave as-is (validate_and_clean will reject it)


def validate_and_clean(found):
    """
    Normalizes common abbreviations/case differences, then removes any field
    whose value is still invalid -- not in its allowed list, or (for age) not
    a realistic number. Invalid fields are dropped so they get asked again.
    Returns (cleaned_dict, rejected_dict) where rejected_dict maps
    field_name -> a short explanation of why it was rejected.
    """
    cleaned = {}
    rejected = {}
    for field, raw_value in found.items():
        value = normalize_value(field, raw_value)

        if field == "age":
            try:
                age_val = float(value)
                if 13 <= age_val <= 100:
                    cleaned[field] = value
                else:
                    rejected[field] = "That doesn't look like a realistic age."
            except (ValueError, TypeError):
                rejected[field] = "That didn't look like a valid age."
        elif field in ALLOWED_VALUES and ALLOWED_VALUES[field] is not None:
            if value in ALLOWED_VALUES[field]:
                cleaned[field] = value
            else:
                rejected[field] = "I didn't quite catch a valid answer for that one."
        else:
            cleaned[field] = value
    return cleaned, rejected


def handle_message(user_message, collected, pending_fields=None):
    """
    Main entry point for one turn of conversation.
    'collected' is a dict that persists across turns (e.g. in Streamlit session state).
    'pending_fields' is the batch of fields we asked about last turn (None on the first turn).
    Returns (response_text, updated_collected, prediction_or_None, new_pending_fields).
    """
    used_fast_path = False
    rejected = {}

    # Fast path: if we're only waiting on ONE specific field, check the raw
    # answer against its allowed values directly -- more reliable than the LLM
    # for short answers ("f", "m", "n"), and skips an API call entirely.
    if pending_fields and len(pending_fields) == 1:
        field = pending_fields[0]
        direct_value = normalize_value(field, user_message.strip())

        is_valid = False
        if field == "age":
            try:
                is_valid = 13 <= float(direct_value) <= 100
            except (ValueError, TypeError):
                is_valid = False
        elif field in ALLOWED_VALUES and ALLOWED_VALUES[field] is not None:
            is_valid = direct_value in ALLOWED_VALUES[field]

        if is_valid:
            collected[field] = direct_value
            used_fast_path = True

    if not used_fast_path:
        parsed = parse_user_input(user_message, pending_fields=pending_fields)
        cleaned_found, rejected = validate_and_clean(parsed["found"])
        collected.update(cleaned_found)

    missing = [f for f in CORE_FIELDS if f not in collected]
    if missing:
        next_batch = missing[:QUESTIONS_PER_BATCH]
        return ask_for_missing(missing, rejected), collected, None, next_batch

    # Fill in reasonable defaults for the fields we didn't actively ask about
    for field, default_value in DEFAULT_VALUES.items():
        collected.setdefault(field, default_value)

    age_scaler, feature_columns = load_artifacts()
    encoded_row = encode_features(collected, feature_columns, age_scaler)

    model, model_name = load_best_model()
    prediction, probability = predict(encoded_row, model)

    response_text = generate_response(collected, prediction, probability)
    response_text = f"Thanks for answering those questions!\n\n{response_text}"
    return response_text, collected, prediction, []


WELCOME_MESSAGE = (
    "Hi! I can estimate whether someone is likely to seek mental health treatment, "
    "based on a model trained on real workplace survey data.\n\n"
    "To get started, tell me a bit about yourself — for example: your age, gender, "
    "whether you work remotely, your company size, and whether you have a family "
    "history of mental illness. I'll ask a few follow-up questions if I need more info."
)


if __name__ == "__main__":
    print(WELCOME_MESSAGE)
    print("\n(type 'quit' to exit, 'show' to see collected answers)\n")
    collected = {}
    pending_fields = None

    while True:
        user_message = input("You: ")
        if user_message.lower() == "quit":
            break
        if user_message.lower() == "show":
            print(f"\nCollected so far: {json.dumps(collected, indent=2)}\n")
            continue

        response_text, collected, prediction, pending_fields = handle_message(
            user_message, collected, pending_fields
        )
        print(f"\nAssistant: {response_text}\n")