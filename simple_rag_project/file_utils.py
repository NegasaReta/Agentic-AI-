
import io
import pypdf
import docx

def read_txt(file_byte_stream):
    """Reads text from a bytes stream (TXT)."""
    return file_byte_stream.read().decode("utf-8")

def read_pdf(file_byte_stream):
    """Extracts text from a PDF file bytes stream."""
    reader = pypdf.PdfReader(file_byte_stream)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)

def read_docx(file_byte_stream):
    """Extracts text from a DOCX file bytes stream."""
    doc = docx.Document(file_byte_stream)
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    return "\n".join(text)

def read_file_content(uploaded_file):
    """
    Dispatcher to read content based on file type.
    uploaded_file is a Streamlit UploadedFile object.
    """
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    # Reset pointer just in case, though streamlit usually handles this
    uploaded_file.seek(0)
    
    if file_type == 'txt':
        return read_txt(uploaded_file)
    elif file_type == 'pdf':
        return read_pdf(uploaded_file)
    elif file_type in ['docx', 'doc']:
        return read_docx(uploaded_file)
    else:
        return f"[Error] Unsupported file type: {file_type}"
