"""
Visualize annotated report.
"""

import spacy
from spacy import displacy

import utils


export_path = utils.derivatives_directory / "annotation.html"

# annotator1["textlength"] = annotator1.report.str.len()
# annotator1.sort_values("textlength", ascending=True).head(30)
annotator1 = utils.load_annotations("annotator1", drop_text=False)
report_id = "report-e74b"
text = annotator1.at[report_id, "report"]
labels = annotator1.at[report_id, "labels"]

# Convert character indices to token indices
nlp = spacy.blank("en")
doc = nlp(text)

char_to_token = {token.idx: token.i for token in doc}
for i in range(len(text)):
    if i not in char_to_token:
        char_to_token[i] = char_to_token[i - 1]

# Update labels with token indices
spans = [
    {
        "start_token": char_to_token[start],
        "end_token": char_to_token[end],
        "label": label.capitalize(),
    }
    for start, end, label in labels
]

palette = {k.capitalize(): v for k, v in utils.get_color_palette().items()}

tokens = [token.text for token in doc]
# tokens = list(doc)

data = {"text": text, "spans": spans, "tokens": tokens}
options = {"colors": palette}

html = displacy.render(data, style="span", manual=True, options=options)

html = html.replace('ltr"', 'ltr; text-align: center; max-width: 6in; margin: 0 auto;"')

with export_path.open("w", encoding="utf-8") as f:
    f.write(html)
