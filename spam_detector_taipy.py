from taipy.gui import Gui, notify
import pickle
import string
import re
import math
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()

# Ensure NLTK data is available at runtime (helpful on fresh environments)
try:
    _ = stopwords.words('english')
    _ = nltk.word_tokenize("ok")
except Exception:
    for pkg in ["punkt", "wordnet", "stopwords"]:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

# Cache resources that are reused across predictions for lower latency
STOPWORDS_SET = set()
try:
    STOPWORDS_SET = set(stopwords.words('english'))
except Exception:
    STOPWORDS_SET = set()

# Precompile regex patterns used in heuristics
URL_RE = re.compile(r'(https?://|www\.)', re.IGNORECASE)
MONEY_RE = re.compile(r'\b(rs|inr|usd|lakh|crore|transfer|send|payment|deposit|cashback|bonus|prize|win|loan|emi)\b', re.IGNORECASE)
BANK_RE = re.compile(r'\b(bank|account|acct|ifsc|upi|card|cvv|otp|pin|kyc|verify|update)\b', re.IGNORECASE)
DIGITS_RE = re.compile(r'\b\d{6,}\b')
URGENT_RE = re.compile(r'\b(urgent|immediate|action required|limited time|expires)\b', re.IGNORECASE)

HAM_RE_LIST = [
    (re.compile(r"\b(payment )?was successful\b", re.IGNORECASE), 0.14),
    (re.compile(r"\bthank you\b", re.IGNORECASE), 0.10),
    (re.compile(r"\bpayment received\b", re.IGNORECASE), 0.14),
    (re.compile(r"\breceipt\b", re.IGNORECASE), 0.12),
    (re.compile(r"\binvoice\b", re.IGNORECASE), 0.10),
    (re.compile(r"\bbill\b", re.IGNORECASE), 0.10),
    (re.compile(r"\border (confirmed|delivered|shipped)\b", re.IGNORECASE), 0.10),
    (re.compile(r"\bbooking (confirmed|successful)\b", re.IGNORECASE), 0.12),
]

ACCOUNT_DOC_RE = re.compile(r"\b(account|invoice|bill|order|ticket)\b", re.IGNORECASE)

def transform_text(text):
    # Lowercase and tokenize once
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    # Remove non-alphanumeric tokens
    alnum_tokens = [tok for tok in tokens if tok.isalnum()]
    # Filter stopwords and punctuation using cached set
    if STOPWORDS_SET:
        filtered_tokens = [tok for tok in alnum_tokens if tok not in STOPWORDS_SET and tok not in string.punctuation]
    else:
        filtered_tokens = [tok for tok in alnum_tokens if tok not in string.punctuation]
    # Lemmatize
    lemmatized = [lemma.lemmatize(tok) for tok in filtered_tokens]
    return " ".join(lemmatized)
with open('vectorizer.pkl','rb') as f:
    tfidf = pickle.load(f)
with open('model.pkl','rb') as f:
    model = pickle.load(f)

intro = "Enter the message:"
dummy = " "
message = " "
h_text = "Click to classify the spam or ham mail/sms"

# Configuration
THRESHOLD = 0.40  # fixed threshold (UI slider removed)
STRICT_MODE = True  # keep strict behavior on by default

page = """
<|text-center|
#
# Email/SMS **Spam**{:.color-primary} Detector
|>

<|{intro}|>

#

<|{dummy}|input|multiline|class_name=fullwidth|label=Enter your message here|>

#

<|text-center|

<|Predict|button|on_action=on_button_action|hover_text = Click to classify the spam or ham mail/sms|center|>

#

<|{message}|input|not active|label= Know the results here!|>




"""

def on_button_action(state):
    notify(state, 'info', f'Results may be inaccurate due to limited dataset. Thanks for using this Application :p')
    state.message = "Rendering results..."
    
    # 1. preprocess
    transformed_sms = transform_text(state.dummy)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict with probability + heuristics
    def get_model_spam_probability():
        # Use predict_proba if available
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(vector_input)[0]
                # assume positive/spam class is 1 if present
                if len(proba) == 2:
                    return float(proba[1])
                # fallback: take max as conservative
                return float(max(proba))
            except Exception:
                pass
        # Use decision_function if available (e.g., LinearSVC) and map via sigmoid
        if hasattr(model, 'decision_function'):
            try:
                df = model.decision_function(vector_input)
                # df can be array-like
                score = float(df[0]) if hasattr(df, '__len__') else float(df)
                # map to (0,1)
                return 1.0 / (1.0 + math.exp(-score))
            except Exception:
                pass
        # Fallback to label prediction (0/1)
        try:
            pred = int(model.predict(vector_input)[0])
            return 1.0 if pred == 1 else 0.0
        except Exception:
            return 0.0

    base_proba = get_model_spam_probability()

    # Heuristic boosts for typical spam cues (URLs, money, bank/account, large numbers)
    raw_text = state.dummy.lower()
    matched_cues = []
    positive_boost = 0.0

    # Use precompiled regex for faster matching
    if URL_RE.search(raw_text):
        matched_cues.append("URL detected")
        positive_boost += 0.14
    if MONEY_RE.search(raw_text):
        matched_cues.append("Money/transfer terms")
        positive_boost += 0.14
    if BANK_RE.search(raw_text):
        matched_cues.append("Banking/KYC terms")
        positive_boost += 0.14
    if DIGITS_RE.search(raw_text):
        matched_cues.append("Long number sequence")
        positive_boost += 0.10
    if URGENT_RE.search(raw_text):
        matched_cues.append("Urgency language")
        positive_boost += 0.08

    # Ham cues: phrases typical of legitimate confirmations/receipts
    ham_reduction = 0.0
    for ham_re, dec in HAM_RE_LIST:
        if ham_re.search(raw_text):
            ham_reduction += dec

    # If long digits appear alongside account/bill/invoice, reduce that positive cue slightly
    if DIGITS_RE.search(raw_text) and ACCOUNT_DOC_RE.search(raw_text):
        ham_reduction += 0.05

    # In strict mode, lightly scale up boosts to prefer catching spam in demos
    boost = positive_boost - ham_reduction
    if boost < 0.0:
        boost = 0.0
    if STRICT_MODE:
        boost *= 1.15

    spam_score = min(1.0, base_proba + boost)

    # 4. Display using fixed threshold (no score shown)
    if spam_score >= THRESHOLD:
        state.message = "Spam"
    else:
        state.message = "Not-Spam"


if __name__ == "__main__":
    app = Gui(page).run(watermark="Taipy Inside :)", use_reloader=True, port="auto")