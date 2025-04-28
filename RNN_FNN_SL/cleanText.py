
from nltk.corpus import stopwords
import re
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower().strip()  # Små bokstaver og fjern whitespace
    words = re.findall(r'\b\w+\b', text)  # Beholder bare ord (bokstaver og tall)
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)
