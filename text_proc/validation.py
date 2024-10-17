import re

def validate_structured_fields(texts):
    validated_texts = []
    for text in texts:
        if re.match(r'^\d{9,10}$', text):
            validated_texts.append(text)
        else:
            validated_texts.append("INVALID")
    return validated_texts
