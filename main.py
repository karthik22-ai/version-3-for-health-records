from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from firebase_admin import credentials, initialize_app, firestore, storage
import google.generativeai as genai
import os
from dotenv import load_dotenv
import io
import mimetypes
from datetime import datetime
import json
import time # Import time for synchronous sleep

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# --- Firebase Initialization ---
FIREBASE_CREDENTIALS_PATH = os.getenv('FIREBASE_CREDENTIALS_PATH')
if FIREBASE_CREDENTIALS_PATH and os.path.exists(FIREBASE_CREDENTIALS_PATH):
    print(f"Attempting Firebase initialization using credentials from: {FIREBASE_CREDENTIALS_PATH}")
    try:
        cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
    except Exception as e:
        print(f"ERROR: Failed to load Firebase credentials from path '{FIREBASE_CREDENTIALS_PATH}': {e}")
        print("Please verify the path and content of your Firebase service account key JSON file.")
        exit("Firebase credentials are required to start the backend. Exiting.") # Exit if explicit path fails
else:
    print("FIREBASE_CREDENTIALS_PATH not found or not set. Attempting default Firebase initialization (for Google Cloud environments).")
    try:
        cred = credentials.ApplicationDefault()
        print("Using Application Default Credentials. This typically works in Google Cloud environments.")
    except Exception as e:
        print(f"ERROR: Failed to get ApplicationDefault credentials: {e}")
        print("Please ensure GOOGLE_APPLICATION_CREDENTIALS is set or FIREBASE_CREDENTIALS_PATH points to a valid service account key for local development.")
        exit("Firebase credentials are required to start the backend. Exiting.") # Exit if default fails

try:
    firebase_app = initialize_app(cred, {
        'projectId': os.getenv('FIREBASE_PROJECT_ID'),
        'storageBucket': os.getenv('FIREBASE_STORAGE_BUCKET')
    })
    db = firestore.client()
    bucket = storage.bucket()
    print("Firebase initialized successfully.")
except Exception as e:
    print(f"ERROR: Firebase initialization failed after credentials loaded: {e}")
    exit("Firebase initialization failed. Exiting.")


# --- Gemini API Configuration ---
# Using GEMINI_API_KEY as per the user's provided file
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not found. AI functionalities will not work.")
    genai.configure(api_key="YOUR_GEMINI_API_KEY_HERE") # Placeholder, will fail if not set
else:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API configured.")

# Use gemini-2.5-flash for both multimodal and structured capabilities
gemini_model = genai.GenerativeModel('gemini-2.5-flash') 

# Define allowed document categories for Gemini
DOCUMENT_CATEGORIES = [
    'Lab Results', 'Prescriptions', 'Radiology', 'Discharge Summaries',
    'Vital Signs', 'Insurance', 'Consultation Notes', 'Other'
]

# --- Helper Functions ---

def get_user_id_from_request():
    """Extracts user ID from the X-User-Id header."""
    user_id = request.headers.get('X-User-Id')
    if not user_id:
        print("Warning: X-User-Id header not found. Using 'anonymous_user'. For production, implement proper token verification.")
        user_id = "anonymous_user"
    return user_id

def get_app_id():
    """Retrieves the app ID from environment variables or uses a default."""
    # Using APP_ID as per user's provided file
    return os.getenv('APP_ID', 'default-app-id')

def get_user_document_collection_path(user_id):
    """Constructs the Firestore collection path for a user's documents."""
    app_id = get_app_id()
    return f'artifacts/{app_id}/users/{user_id}/documents'

def get_health_profiles_collection_path(user_id):
    """Constructs the Firestore collection path for a user's health profiles."""
    app_id = get_app_id()
    return f'artifacts/{app_id}/users/{user_id}/health_profiles'

def get_family_group_collection_path():
    """Constructs the Firestore collection path for family groups."""
    app_id = get_app_id()
    return f'artifacts/{app_id}/family_groups'

def get_family_members(family_group_id):
    """Fetches members of a given family group synchronously."""
    app_id = get_app_id()
    members = []
    if family_group_id:
        try:
            group_doc_ref = db.collection(f'artifacts/{app_id}/family_groups').document(family_group_id)
            group_doc = group_doc_ref.get() # Synchronous Firestore call
            if group_doc.exists:
                members = group_doc.to_dict().get('members', [])
                print(f"Found {len(members)} members for family group {family_group_id}.")
            else:
                print(f"Family group {family_group_id} not found.")
        except Exception as e:
            print(f"Error getting family members for group {family_group_id}: {e}")
    return members

def generate_full_ai_analysis(document_content):
    """
    Uses Gemini to perform a deep analysis of a medical document.
    Extracts structured data including vitals, diagnosis, medications, etc.
    This is a synchronous function.
    """
    if not GEMINI_API_KEY:
        print("Skipping AI analysis: GEMINI_API_KEY is not set.")
        return {
            "category": "Other", "summary": "AI analysis skipped: API key missing.",
            "doctorName": "N/A", "change_analysis": "AI analysis skipped: API key missing.",
            "hospital_name": "N/A", "date_of_visit": None, "diagnosis": "N/A",
            "medications": [], "problem": "N/A", "blood_pressure": "N/A",
            "blood_sugar": "N/A", "weight": "N/A"
        }

    try:
        response_schema = {
            "type": "OBJECT",
            "properties": {
                "category": {"type": "STRING", "enum": DOCUMENT_CATEGORIES},
                "summary": {"type": "STRING"},
                "doctorName": {"type": "STRING"},
                "hospital_name": {"type": "STRING"},
                "date_of_visit": {"type": "STRING", "description": "The primary date of the visit/report in YYYY-MM-DD format."},
                "problem": {"type": "STRING", "description": "A short, clear title for the main health issue or purpose of this document (e.g., 'Annual Check-up', 'Flu Symptoms', 'Follow-up for Diabetes')."},
                "diagnosis": {"type": "STRING", "description": "The specific diagnosis mentioned in the document."},
                "medications": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"}
                },
                "blood_pressure": {"type": "STRING", "description": "Blood pressure reading if present (e.g., '120/80 mmHg')."},
                "blood_sugar": {"type": "STRING", "description": "Blood sugar reading if present (e.g., '95 mg/dL')."},
                "weight": {"type": "STRING", "description": "Patient's weight if present (e.g., '165 lbs' or '75 kg')."},
                "change_analysis": {"type": "STRING", "description": "Interpretation of the findings or changes."},
            },
            "required": ["category", "summary", "doctorName", "hospital_name", "date_of_visit", "problem", "diagnosis", "medications", "blood_pressure", "blood_sugar", "weight", "change_analysis"]
        }

        prompt = f"""
        Analyze the following medical document text. Extract the specified information precisely and return it in a single JSON object.

        Document Text:
        ---
        {document_content[:8000]}
        ---

        Please extract the following details:
        1.  **category**: Categorize into one of: {', '.join(DOCUMENT_CATEGORIES)}.
        2.  **summary**: A concise summary of the document.
        3.  **doctorName**: The primary doctor's name. If none, use "N/A".
        4.  **hospital_name**: The name of the hospital or clinic. If none, use "N/A".
        5.  **date_of_visit**: The main date of the consultation, lab test, or report. **Format strictly as YYYY-MM-DD**. If no date is found, return null.
        6.  **problem**: A short, clear title for the main health issue (e.g., 'Annual Check-up', 'Follow-up for Diabetes').
        7.  **diagnosis**: The specific diagnosis given. If none, use "N/A".
        8.  **medications**: A list of all prescribed medications. If none, return an empty list [].
        9.  **blood_pressure**: Find the blood pressure reading. If not present, use "N/A".
        10. **blood_sugar**: Find the blood sugar/glucose reading. If not present, use "N/A".
        11. **weight**: Find the patient's weight. If not present, use "N/A".
        12. **change_analysis**: A brief interpretation of the document's findings.
        """

        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            generation_config={"response_mime_type": "application/json", "response_schema": response_schema}
        )
        
        retries = 0
        max_retries = 5
        while retries < max_retries:
            try:
                response = model.generate_content(prompt)
                if response.candidates and response.candidates[0].content:
                    return json.loads(response.candidates[0].content.parts[0].text)
                else:
                    raise ValueError("Gemini API returned no content.")
            except Exception as e:
                retries += 1
                wait_time = 2 ** retries
                print(f"Gemini API call failed (attempt {retries}/{max_retries}): {e}. Retrying in {wait_time} seconds...")
                if retries < max_retries:
                    time.sleep(wait_time)
                else:
                    raise
    except Exception as e:
        print(f"Error during Gemini AI analysis: {e}")
        return {
            "category": "Other", "summary": f"AI analysis failed: {e}",
            "doctorName": "N/A", "change_analysis": f"AI analysis failed: {e}",
            "hospital_name": "N/A", "date_of_visit": None, "diagnosis": "N/A",
            "medications": [], "problem": "N/A", "blood_pressure": "N/A",
            "blood_sugar": "N/A", "weight": "N/A"
        }

def extract_text_from_file(file_stream, mime_type, file_name):
    """
    Extracts text content from plain text files, or generates a simulated text
    for PDF and image files using Gemini.
    Returns the extracted/simulated text as a string.
    This is a synchronous function.
    """
    text_content = ""
    if mime_type == 'text/plain':
        try:
            text_content = file_stream.read().decode('utf-8')
            print(f"Extracted text from plain text file: {file_name}")
        except Exception as e:
            print(f"Error reading text file: {e}")
            return None
    elif mime_type in ['application/pdf', 'image/jpeg', 'image/png']:
        if not GEMINI_API_KEY:
            print("Skipping text extraction for image/PDF: GEMINI_API_KEY is not set.")
            return f"Simulated content for {file_name} ({mime_type}). AI analysis will be limited."

        file_stream.seek(0)
        file_bytes = file_stream.read()

        image_part = {
            "mime_type": mime_type,
            "data": file_bytes
        }
        
        text_extraction_prompt_parts = [
            image_part,
            "Extract all readable text from this document. Provide only the extracted text, no other conversational remarks or formatting. If no text is clearly visible, state 'No readable text found'."
        ]
        print(f"Attempting text extraction for {file_name} using Gemini (multimodal)...")
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(text_extraction_prompt_parts) # Synchronous call
            extracted_text = response.candidates[0].content.parts[0].text.strip()
            if extracted_text == "No readable text found":
                print(f"Gemini reported no readable text for {file_name}.")
                text_content = ""
            else:
                text_content = extracted_text
            print(f"Gemini Text Extracted for {file_name} (first 100 chars): {text_content[:100]}...")
        except Exception as e:
            print(f"Error during Gemini text extraction for {file_name}: {e}")
            text_content = f"Failed to extract text using AI for {file_name}. Error: {e}"
    else:
        print(f"Unsupported MIME type for text extraction: {mime_type}")
        return None
    return text_content


# --- Backend Endpoints ---

@app.route('/upload', methods=['POST'])
def upload_document():
    """
    Handles document upload, stores original in Firebase Storage,
    performs text extraction and categorization with Gemini,
    and stores metadata/digital copy in Firestore.
    This endpoint can now accept either a single file under 'file'
    or multiple files under 'files'.
    """
    uploaded_files = []

    if 'files' in request.files:
        uploaded_files = request.files.getlist('files')
        print(f"Detected {len(uploaded_files)} files from 'files' field.")
    elif 'file' in request.files:
        single_file = request.files['file']
        if single_file.filename != '':
            uploaded_files.append(single_file)
            print("Detected single file from 'file' field.")
    
    if not uploaded_files:
        return jsonify({'error': 'No files selected for upload or invalid form data.'}), 400

    user_id = get_user_id_from_request()
    app_id = get_app_id()
    
    processed_results = []

    for file in uploaded_files:
        if file.filename == '':
            continue

        original_filename = file.filename
        file_extension = os.path.splitext(original_filename)[1]
        unique_filename = f"{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{os.urandom(4).hex()}{file_extension}"
        blob_path = f"artifacts/{app_id}/users/{user_id}/original_documents/{unique_filename}"
        file_error = None

        try:
            file_content = file.read()
            file_mime_type = file.content_type
            print(f"Processing file: {original_filename}, MIME Type: {file_mime_type}")
            
            blob = bucket.blob(blob_path)
            blob.upload_from_string(file_content, content_type=file_mime_type) # Synchronous
            blob.make_public() # Synchronous
            original_file_url = blob.public_url
            print(f"File uploaded to Storage: {original_file_url}")

            file_stream = io.BytesIO(file_content)
            extracted_text_for_digital_copy = extract_text_from_file(file_stream, file_mime_type, original_filename) # Synchronous call

            if extracted_text_for_digital_copy:
                ai_analysis = generate_full_ai_analysis(extracted_text_for_digital_copy) # Synchronous call
                print(f"AI Analysis completed for {original_filename}. Category: {ai_analysis.get('category')}")
            else:
                ai_analysis = {} # Empty dict if no text
                file_error = "No text extracted for AI analysis."
                print(f"Warning: No text extracted for AI analysis for {original_filename}.")

            # Construct document data from AI analysis
            doc_data_to_save = {
                'name': original_filename,
                'type': file_mime_type,
                'original_url': original_file_url,
                'digital_copy_content': extracted_text_for_digital_copy,
                'size': len(file_content),
                'timestamp': firestore.SERVER_TIMESTAMP,
                'ownerId': user_id,
                'category': ai_analysis.get('category', 'Other'),
                'summary': ai_analysis.get('summary', ''),
                'doctorName': ai_analysis.get('doctorName', 'N/A'),
                'hospital_name': ai_analysis.get('hospital_name', 'N/A'),
                'date_of_visit': ai_analysis.get('date_of_visit'),
                'problem': ai_analysis.get('problem', 'N/A'),
                'diagnosis': ai_analysis.get('diagnosis', 'N/A'),
                'medications': ai_analysis.get('medications', []),
                'blood_pressure': ai_analysis.get('blood_pressure', 'N/A'),
                'blood_sugar': ai_analysis.get('blood_sugar', 'N/A'),
                'weight': ai_analysis.get('weight', 'N/A'),
                'change_analysis': ai_analysis.get('change_analysis', '')
            }

            doc_ref = db.collection(get_user_document_collection_path(user_id))
            _, doc_added_ref = doc_ref.add(doc_data_to_save) # Synchronous call
            print(f"Document metadata saved to Firestore with ID: {doc_added_ref.id}")

            result_entry = {
                'filename': original_filename,
                'status': 'success',
                'documentId': doc_added_ref.id,
                'error': file_error,
                **doc_data_to_save # Add all saved data to the response
            }
            del result_entry['timestamp'] # Not needed in JSON response
            processed_results.append(result_entry)

        except Exception as e:
            print(f"CRITICAL ERROR during processing of {original_filename}: {e}")
            processed_results.append({
                'filename': original_filename,
                'status': 'failed',
                'error': str(e)
            })

    return jsonify({'message': 'Multiple documents processed', 'results': processed_results}), 200

@app.route('/documents', methods=['GET'])
def get_documents():
    """
    Retrieves all document metadata for the current user from Firestore.
    Includes documents from linked family members.
    """
    user_id = get_user_id_from_request()
    app_id = get_app_id()

    documents = []
    print(f"Fetching documents for user: {user_id} in app: {app_id}")
    try:
        # Get user's own documents
        user_docs_ref = db.collection(f'artifacts/{app_id}/users/{user_id}/documents')
        user_docs_snapshots = user_docs_ref.stream() # Use stream for iteration
        print(f"Found documents for user {user_id}.")
        for doc_snapshot in user_docs_snapshots:
            doc_data = doc_snapshot.to_dict()
            doc_data['id'] = doc_snapshot.id
            # Safely convert timestamp to ISO format string
            timestamp_val = doc_data.get('timestamp')
            if isinstance(timestamp_val, datetime):
                doc_data['timestamp'] = timestamp_val.isoformat()
            elif isinstance(timestamp_val, str):
                doc_data['timestamp'] = timestamp_val # Already a string
            else:
                doc_data['timestamp'] = None # Handle other unexpected types
            documents.append(doc_data)

        # Get family group documents
        family_groups_ref = db.collection(f'artifacts/{app_id}/family_groups')
        user_family_groups = family_groups_ref.where('members', 'array_contains', {'userId': user_id}).stream()
        print(f"Found family groups for user {user_id}.")

        for group_doc in user_family_groups:
            group_data = group_doc.to_dict()
            for member in group_data.get('members', []):
                member_id = member.get('userId')
                if member_id != user_id: # Don't re-fetch own documents
                    print(f"Fetching documents for family member: {member_id}")
                    member_docs_ref = db.collection(f'artifacts/{app_id}/users/{member_id}/documents')
                    member_docs_snapshots = member_docs_ref.stream()
                    for doc_snapshot in member_docs_snapshots:
                        doc_data = doc_snapshot.to_dict()
                        doc_data['id'] = doc_snapshot.id
                        timestamp_val = doc_data.get('timestamp')
                        if isinstance(timestamp_val, datetime):
                            doc_data['timestamp'] = timestamp_val.isoformat()
                        elif isinstance(timestamp_val, str):
                            doc_data['timestamp'] = timestamp_val
                        else:
                            doc_data['timestamp'] = None
                        if doc_data['id'] not in [d['id'] for d in documents]:
                            documents.append(doc_data)
                            print(f"Added family document: {doc_data.get('name')} from {member_id}")

        documents.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        print(f"Total documents retrieved: {len(documents)}")
        return jsonify(documents), 200
    except Exception as e:
        print(f"Error fetching documents: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/documents/<document_id>/download/original', methods=['GET'])
def download_original_document(document_id):
    """
    Downloads the original document file from Firebase Storage.
    """
    user_id = get_user_id_from_request()
    app_id = get_app_id()

    doc_ref = db.collection(f'artifacts/{app_id}/users/{user_id}/documents').document(document_id)
    print(f"Attempting to download original document {document_id} for user {user_id}")
    try:
        doc_snapshot = doc_ref.get() # Synchronous Firestore call
        if not doc_snapshot.exists:
            print(f"Document {document_id} not found in user's collection.")
            return jsonify({'error': 'Document not found'}), 404

        doc_data = doc_snapshot.to_dict()
        original_url = doc_data.get('original_url')
        file_name = doc_data.get('name')
        mime_type = doc_data.get('type', 'application/octet-stream')

        if not original_url:
            print(f"Original URL missing for document {document_id}.")
            return jsonify({'error': 'Original file URL not found for this document'}), 404

        path_segments = original_url.split(f'https://storage.googleapis.com/{bucket.name}/')
        if len(path_segments) < 2:
            print(f"Invalid original file URL format for document {document_id}: {original_url}")
            return jsonify({'error': 'Invalid original file URL format'}), 500
        
        blob_name = path_segments[1]
        blob = bucket.blob(blob_name)
        print(f"Downloading blob: {blob_name}")
        file_content = blob.download_as_bytes() # Synchronous
        print(f"Successfully downloaded {file_name} from Storage.")

        return send_file(
            io.BytesIO(file_content),
            mimetype=mime_type,
            as_attachment=False,
            download_name=file_name
        )
    except Exception as e:
        print(f"Error downloading original document {document_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/documents/<document_id>/download/digital_copy', methods=['GET'])
def download_digital_copy(document_id):
    """
    Downloads the processed digital copy content (entire extracted text) as a .txt file.
    """
    user_id = get_user_id_from_request()
    app_id = get_app_id()

    doc_ref = db.collection(f'artifacts/{app_id}/users/{user_id}/documents').document(document_id)
    print(f"Attempting to download digital copy for document {document_id} for user {user_id}")
    try:
        doc_snapshot = doc_ref.get() # Synchronous Firestore call
        if not doc_snapshot.exists:
            print(f"Document {document_id} not found in user's collection for digital copy.")
            return jsonify({'error': 'Document not found'}), 404

        doc_data = doc_snapshot.to_dict()
        digital_copy_content = doc_data.get('digital_copy_content', '')
        original_file_name = doc_data.get('name', 'digital_copy')
        base_name = os.path.splitext(original_file_name)[0]
        digital_copy_filename = f"{base_name}_extracted_text.txt"

        if not digital_copy_content:
            print(f"Digital copy content not available for document {document_id}.")
            return jsonify({'error': 'Digital copy content not available for this document'}), 404

        print(f"Serving digital copy for {document_id} as {digital_copy_filename}.")
        return send_file(
            io.BytesIO(digital_copy_content.encode('utf-8')),
            mimetype='text/plain',
            as_attachment=False,
            download_name=digital_copy_filename
        )
    except Exception as e:
        print(f"Error downloading digital copy for document {document_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/documents/<document_id>', methods=['DELETE'])
def delete_document(document_id):
    """
    Deletes a document from Firestore and its corresponding file from Firebase Storage.
    """
    user_id = get_user_id_from_request()
    app_id = get_app_id()

    doc_ref = db.collection(f'artifacts/{app_id}/users/{user_id}/documents').document(document_id)
    print(f"Attempting to delete document {document_id} for user {user_id}.")
    try:
        doc_snapshot = doc_ref.get() # Synchronous Firestore call
        if not doc_snapshot.exists:
            print(f"Document {document_id} not found for deletion.")
            return jsonify({'error': 'Document not found'}), 404

        doc_data = doc_snapshot.to_dict()
        original_url = doc_data.get('original_url')

        if original_url:
            try:
                path_segments = original_url.split(f'https://storage.googleapis.com/{bucket.name}/')
                if len(path_segments) > 1:
                    blob_name = path_segments[1]
                    blob = bucket.blob(blob_name)
                    blob.delete() # Synchronous
                    print(f"Deleted file from storage: {blob_name}")
                else:
                    print(f"Warning: Could not extract blob name from URL: {original_url} for deletion.")
            except Exception as e:
                print(f"Warning: Could not delete file from storage ({original_url}) for document {document_id}: {e}")

        doc_ref.delete() # Synchronous Firestore call
        print(f"Document {document_id} metadata deleted from Firestore.")
        return jsonify({'message': 'Document deleted successfully'}), 200
    except Exception as e:
        print(f"Error deleting document {document_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analytics', methods=['POST'])
def generate_analytics_for_selected_documents():
    """
    Generates an analytics report based on a list of provided document IDs.
    It will also trigger AI analysis for documents that haven't been analyzed or need re-analysis.
    Expected JSON body: {"document_ids": ["id1", "id2", ...]}
    """
    user_id = get_user_id_from_request()
    if not user_id:
        return jsonify({'error': 'Unauthorized: User ID missing.'}), 401

    data = request.get_json()
    selected_document_ids = data.get('document_ids', [])

    if not selected_document_ids:
        print("No specific document IDs provided for analytics. Fetching all user/family documents.")
        all_docs_response, status_code = get_documents()
        if status_code != 200:
            return all_docs_response, status_code
        
        all_docs_data = json.loads(all_docs_response.data)
        selected_document_ids = [doc['id'] for doc in all_docs_data]
        if not selected_document_ids:
            return jsonify({'message': 'No documents found to generate analytics for.'}), 200


    app_id = get_app_id()
    
    total_documents = 0
    total_size_bytes = 0
    documents_by_category = {category: 0 for category in DOCUMENT_CATEGORIES}
    documents_by_category['Other'] = 0
    detailed_report_analytics = []
    
    is_family_analytics = False
    print(f"Generating analytics for {len(selected_document_ids)} selected documents.")

    try:
        selected_docs_data_from_db = []
        # This part needs to be more robust to check all potential user collections
        # For simplicity, we assume IDs are unique and we can find them.
        # A more robust solution might involve checking the current user's collection, then family members' collections.
        all_user_ids_to_check = [user_id]
        family_groups_ref = db.collection(get_family_group_collection_path())
        user_family_groups = family_groups_ref.where('members', 'array_contains', {'userId': user_id}).stream()
        for group in user_family_groups:
            for member in group.to_dict().get('members', []):
                if member['userId'] not in all_user_ids_to_check:
                    all_user_ids_to_check.append(member['userId'])
                    is_family_analytics = True

        for uid in all_user_ids_to_check:
            for doc_id in selected_document_ids:
                doc_ref = db.collection(get_user_document_collection_path(uid)).document(doc_id)
                doc_snapshot = doc_ref.get()
                if doc_snapshot.exists:
                    doc_data = doc_snapshot.to_dict()
                    doc_data['id'] = doc_snapshot.id
                    if doc_data['id'] not in [d['id'] for d in selected_docs_data_from_db]:
                         selected_docs_data_from_db.append(doc_data)


        for doc_data in selected_docs_data_from_db:
            doc_id = doc_data['id']
            
            needs_analysis = not all(doc_data.get(key) for key in ['summary', 'problem', 'diagnosis', 'date_of_visit'])

            if needs_analysis and doc_data.get('digital_copy_content'):
                print(f"Re-analyzing document {doc_data.get('name')} (ID: {doc_id}) for analytics.")
                ai_analysis = generate_full_ai_analysis(doc_data['digital_copy_content'])
                
                owner_id_for_update = doc_data.get('ownerId', user_id)
                doc_to_update_ref = db.collection(get_user_document_collection_path(owner_id_for_update)).document(doc_id)
                update_data = {**ai_analysis, 'last_analyzed': datetime.utcnow().isoformat()}
                doc_to_update_ref.update(update_data)
                doc_data.update(update_data)
            
            total_documents += 1
            total_size_bytes += doc_data.get('size', 0)
            
            category = doc_data.get('category', 'Other')
            documents_by_category[category] = documents_by_category.get(category, 0) + 1

            # Append full data for frontend timeline and modals
            detailed_report_analytics.append(doc_data)

        total_size_kb = round(total_size_bytes / 1024, 2) if total_size_bytes > 0 else 0
        
        overall_summary_and_relation = "No overall summary generated."
        if selected_docs_data_from_db:
            combined_content_for_overall_summary = [
                (f"--- Document: {doc.get('name')} (Date: {doc.get('date_of_visit')}, Problem: {doc.get('problem')}) ---\n"
                 f"Diagnosis: {doc.get('diagnosis')}\n"
                 f"Summary: {doc.get('summary')}\n"
                 f"Medications: {', '.join(doc.get('medications', []))}\n"
                 f"Vitals: BP: {doc.get('blood_pressure')}, Sugar: {doc.get('blood_sugar')}, Weight: {doc.get('weight')}\n")
                for doc in selected_docs_data_from_db if doc.get('digital_copy_content')
            ]
            
            if combined_content_for_overall_summary:
                overall_prompt = (
                    f"Analyze the following collection of medical document summaries for a patient. "
                    f"Provide a comprehensive health overview. Synthesize the information to identify trends, relationships between diagnoses and medications, and the overall health trajectory. "
                    f"Highlight any chronic conditions, recent acute issues, and conflicting information.\n\n"
                    f"Combined Document Analyses:\n" + "\n\n".join(combined_content_for_overall_summary)
                )
                try:
                    overall_summary_response = gemini_model.generate_content(overall_prompt)
                    overall_summary_and_relation = overall_summary_response.text
                except Exception as e:
                    print(f"Error generating overall summary with Gemini: {e}")
                    overall_summary_and_relation = "Failed to generate comprehensive overall summary."

        analytics_report = {
            'total_documents': total_documents,
            'total_size_kb': total_size_kb,
            'documents_by_category': documents_by_category,
            'last_updated': datetime.utcnow().isoformat(),
            'is_family_analytics': is_family_analytics,
            'detailed_report_analytics': detailed_report_analytics,
            'overall_summary': overall_summary_and_relation
        }
        print(f"Analytics report generated. Total documents: {total_documents}")
        return jsonify(analytics_report), 200

    except Exception as e:
        print(f"Error generating analytics report: {e}")
        return jsonify({'error': str(e)}), 500


# --- Health Tracker Endpoints (Removed as logic is now integrated with documents) ---
# The frontend will derive health tracker data directly from the /documents endpoint


# --- Family Endpoints ---

@app.route('/family/status', methods=['GET'], endpoint='get_family_status_endpoint')
def get_family_status():
    """
    Retrieves the current user's family group status.
    Returns the familyGroupId and a list of members in the group.
    """
    user_id = get_user_id_from_request()
    if not user_id:
        return jsonify({'error': 'Unauthorized: User ID missing.'}), 401

    print(f"Checking family status for user: {user_id}")
    try:
        app_id = get_app_id()
        user_doc_ref = db.collection(f'artifacts/{app_id}/users').document(user_id)
        user_doc = user_doc_ref.get() # Synchronous Firestore call
        family_group_id = user_doc.to_dict().get('familyGroupId') if user_doc.exists else None

        members = []
        if family_group_id:
            print(f"User {user_id} is part of family group: {family_group_id}")
            family_groups_ref = db.collection(get_family_group_collection_path())
            group_doc = family_groups_ref.document(family_group_id).get() # Synchronous Firestore call
            if group_doc.exists:
                members = group_doc.to_dict().get('members', [])
                print(f"Family group {family_group_id} has {len(members)} members.")
            else:
                print(f"Family group {family_group_id} not found in Firestore, clearing user's link.")
                user_doc_ref.update({'familyGroupId': firestore.DELETE_FIELD})
                user_name = user_doc.to_dict().get('name', user_id) if user_doc.exists else user_id
                members.append({'userId': user_id, 'name': user_name})
                family_group_id = None # Reset
        else:
            user_name = user_doc.to_dict().get('name', user_id) if user_doc.exists else user_id
            members.append({'userId': user_id, 'name': user_name})
            print(f"User {user_id} is not currently in a family group.")

        return jsonify({'familyGroupId': family_group_id, 'members': members}), 200
    except Exception as e:
        print(f"Error fetching family status: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/family/link-member', methods=['POST'])
def link_family_member():
    """
    Links the current user with another user to form or join a family group.
    """
    user_id = get_user_id_from_request()
    if not user_id:
        return jsonify({'error': 'Unauthorized: User ID missing.'}), 401

    data = request.get_json()
    target_user_id = data.get('targetUserId')
    target_user_name = data.get('targetUserName', target_user_id)

    print(f"Attempting to link user {user_id} with {target_user_id} (Name: {target_user_name})")

    if not target_user_id:
        return jsonify({'error': 'Target User ID is required.'}), 400
    if user_id == target_user_id:
        return jsonify({'error': 'Cannot link to your own account.'}), 400

    try:
        app_id = get_app_id()
        family_groups_ref = db.collection(get_family_group_collection_path())

        # Check existing groups for current user
        current_user_groups_snapshots = list(family_groups_ref.where('members', 'array_contains', {'userId': user_id}).stream())
        current_user_group_id = current_user_groups_snapshots[0].id if current_user_groups_snapshots else None
        current_user_group_data = current_user_groups_snapshots[0].to_dict() if current_user_groups_snapshots else None

        # Check existing groups for target user
        target_user_groups_snapshots = list(family_groups_ref.where('members', 'array_contains', {'userId': target_user_id}).stream())
        target_user_group_id = target_user_groups_snapshots[0].id if target_user_groups_snapshots else None
        target_user_group_data = target_user_groups_snapshots[0].to_dict() if target_user_groups_snapshots else None

        user_doc_ref = db.collection(f'artifacts/{app_id}/users').document(user_id)
        target_doc_ref = db.collection(f'artifacts/{app_id}/users').document(target_user_id)
        
        # Ensure user names are updated/created
        if not user_doc_ref.get().exists or not user_doc_ref.get().to_dict().get('name'):
            user_doc_ref.set({'name': user_id}, merge=True)
        
        if not target_doc_ref.get().exists or not target_doc_ref.get().to_dict().get('name'):
            target_doc_ref.set({'name': target_user_name}, merge=True)

        if current_user_group_id and target_user_group_id:
            if current_user_group_id == target_user_group_id:
                return jsonify({'message': 'Users are already in the same family group.', 'familyGroupId': current_user_group_id, 'members': current_user_group_data.get('members', [])}), 200
            else: # Merge groups
                current_members = current_user_group_data.get('members', [])
                target_members = target_user_group_data.get('members', [])
                new_members = list({m['userId']: m for m in current_members + target_members}.values())
                
                for member in target_members:
                    db.collection(f'artifacts/{app_id}/users').document(member['userId']).update({'familyGroupId': current_user_group_id})
                
                family_groups_ref.document(current_user_group_id).update({'members': new_members})
                family_groups_ref.document(target_user_group_id).delete()
                new_group_id = current_user_group_id
        
        elif current_user_group_id:
            family_group_ref = family_groups_ref.document(current_user_group_id)
            family_group_ref.update({'members': firestore.ArrayUnion([{'userId': target_user_id, 'name': target_user_name}])})
            target_doc_ref.set({'familyGroupId': current_user_group_id}, merge=True)
            new_group_id = current_user_group_id
        
        elif target_user_group_id:
            family_group_ref = family_groups_ref.document(target_user_group_id)
            current_user_name = user_doc_ref.get().to_dict().get('name', user_id)
            family_group_ref.update({'members': firestore.ArrayUnion([{'userId': user_id, 'name': current_user_name}])})
            user_doc_ref.set({'familyGroupId': target_user_group_id}, merge=True)
            new_group_id = target_user_group_id
            
        else: # Create new group
            new_group_ref = family_groups_ref.document()
            current_user_name = user_doc_ref.get().to_dict().get('name', user_id)
            members = [{'userId': user_id, 'name': current_user_name}, {'userId': target_user_id, 'name': target_user_name}]
            new_group_ref.set({'members': members, 'createdAt': firestore.SERVER_TIMESTAMP})
            user_doc_ref.set({'familyGroupId': new_group_ref.id}, merge=True)
            target_doc_ref.set({'familyGroupId': new_group_ref.id}, merge=True)
            new_group_id = new_group_ref.id
        
        final_group_doc = family_groups_ref.document(new_group_id).get()
        members_in_group = final_group_doc.to_dict().get('members', []) if final_group_doc.exists else []
        return jsonify({'message': 'Family member linked successfully', 'familyGroupId': new_group_id, 'members': members_in_group}), 200

    except Exception as e:
        print(f"Error linking family member: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/family/leave', methods=['POST'])
def leave_family_group():
    """
    Allows the current user to leave their family group.
    """
    user_id = get_user_id_from_request()
    if not user_id:
        return jsonify({'error': 'Unauthorized: User ID missing.'}), 401

    print(f"User {user_id} attempting to leave family group.")
    try:
        app_id = get_app_id()
        user_doc_ref = db.collection(f'artifacts/{app_id}/users').document(user_id)
        user_doc_snapshot = user_doc_ref.get()
        current_family_group_id = user_doc_snapshot.to_dict().get('familyGroupId') if user_doc_snapshot.exists else None

        if not current_family_group_id:
            return jsonify({'message': 'Not currently in a family group.'}), 200

        family_group_ref = db.collection(get_family_group_collection_path()).document(current_family_group_id)
        family_group_doc = family_group_ref.get()

        if family_group_doc.exists:
            current_members = family_group_doc.to_dict().get('members', [])
            user_name = user_doc_snapshot.to_dict().get('name', user_id)
            
            # Use a dictionary for the member object to ensure exact match for removal
            member_to_remove = {'userId': user_id, 'name': user_name}
            
            # Update the group by removing the member
            family_group_ref.update({'members': firestore.ArrayRemove([member_to_remove])})
            
            # Check if group is now empty
            updated_group_doc = family_group_ref.get()
            if not updated_group_doc.to_dict().get('members'):
                family_group_ref.delete()
                print(f"Family group {current_family_group_id} deleted as it became empty.")
        
        user_doc_ref.update({'familyGroupId': firestore.DELETE_FIELD})
        print(f"User {user_id}'s familyGroupId cleared.")
        return jsonify({'message': 'Successfully left family group.'}), 200

    except Exception as e:
        print(f"Error unlinking from family group: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)








