import spacy

nlp = spacy.load("en_core_web_sm")

def text_to_iob(text_data):
    iob_data = []
    for sentence in text_data:
        doc = nlp(" ".join(sentence))
        iob_sentence = []
        for token in doc:
            if token.ent_type_ != "":
                if token.i == 0 or doc[token.i-1].ent_type_ != token.ent_type_:
                    iob_tag = "B-" + token.ent_type_
                else:
                    iob_tag = "I-" + token.ent_type_
            else:
                iob_tag = "O"
            iob_sentence.append((token.text, iob_tag))
            
            # Generate "I" tag for named entities that span across multiple tokens
            if token.ent_iob_ == "B" and token.i < len(doc) - 1 and doc[token.i+1].ent_iob_ == "I":
                for next_token in doc[token.i+1:]:
                    if next_token.ent_iob_ == "I":
                        iob_sentence.append((next_token.text, "I-" + next_token.ent_type_))
                    else:
                        break
            
        iob_data.append(iob_sentence)
    return iob_data
text = [["My", "name", "is", "Ahsen", "."], ["Ahsen", "owns", "a", "BMW", "F80", "M3"]]


iob_text = text_to_iob(text)

print(iob_text)
