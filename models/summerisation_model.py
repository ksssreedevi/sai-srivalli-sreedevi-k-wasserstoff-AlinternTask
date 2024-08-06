from transformers import pipeline

summarizer = pipeline("summarization")

def summarize_attributes(object_data):
    summaries = []
    for obj in object_data:
        text_to_summarize = f"Description: {obj['description']}. Extracted Text: {obj.get('extracted_text', 'N/A')}"
        summary = summarizer(text_to_summarize, max_length=50, min_length=25, do_sample=False)
        obj['summary'] = summary[0]['summary_text']
        summaries.append(obj['summary'])
    return object_data
