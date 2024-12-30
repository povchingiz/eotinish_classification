import os, json, binascii
from collections import defaultdict
#from Levenshtein import distance as levenshtein_distance
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from huggingface_hub import login, hf_hub_download

# Hugging Face login
def hf_login(hf_token):
    """Log in to Hugging Face API."""
    try:
        login(token=hf_token)
        print("Logged in to Hugging Face successfully!")
    except Exception as e:
        print(f"Error logging in: {e}")

# Load labels dictionary from Hugging Face repository
def load_labels_dict(repo_id, filename, hf_token):
    """Load labels dictionary from Hugging Face repository."""
    try:
        labels_dict_file = hf_hub_download(repo_id=repo_id, filename=filename, token=hf_token)
        with open(labels_dict_file, "r") as f:
            labels_dict = json.load(f)
        labels_dict = {int(key): value for key, value in labels_dict.items()}
        print("Labels dictionary loaded successfully.")
        return labels_dict
    except Exception as e:
        print(f"Error loading labels dictionary: {e}")
        return {}

# Password encryption function
def password_encrypt(password, plaintext):
    """Encrypt plaintext with a password using AES-CFB encryption."""
    salt = os.urandom(16)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = kdf.derive(password.encode())
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv))
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext.encode()) + encryptor.finalize()
    return binascii.hexlify(ciphertext).decode(), binascii.hexlify(salt).decode(), binascii.hexlify(iv).decode()

# Set up the NER pipeline using Hugging Face model
def setup_ner_pipeline(model_repo, hf_token):
    """Initialize and return a Hugging Face NER pipeline."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_repo, token=hf_token)
        model = AutoModelForTokenClassification.from_pretrained(model_repo, token=hf_token)
        ner_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        return ner_pipeline
    except Exception as e:
        print(f"Failed to set up NER pipeline: {e}")
        return None

# Apply NER model and mark entities with unique identifiers, combining consecutive tokens of the same tag
def apply_model_and_encrypt(text, ner_pipeline, labels_dict, password, tags_to_mask=1, threshold=2):
    """
    Apply NER model to text, assign unique identifiers for detected entities,
    and return a marked sentence and dictionary of entities.
    """
    if tags_to_mask == 1:
        tags_to_mask = ['ART', 'CARDINAL', 'CONTACT', 'DATE','FACILITY', 'GPE', 'LAW', 'LOCATION', 'MISCELLANEOUS','MONEY','NORP','ORDINAL','ORGANISATION','PERCENTAGE','PERSON','POSITION','PRODUCT','PROJECT','QUANTITY','TIME','ACCOUNT','CARD','COMPANY','ADDRESS','IIN','BIN','PHONE','MAIL','WEB']
    else:
        pass
    predictions = ner_pipeline(text)
    processed_predictions = []
    entity_count = defaultdict(int)
    entity_identifiers = {}  # Dictionary to store unique identifiers for entities

    current_entity_tokens = []  # Accumulate tokens for a multi-token entity
    current_label = None  # Track the label of the current entity

    for pred in predictions:
        label_key = pred.get("entity_group", pred.get("entity", None))
        word = pred["word"]

        if label_key is None:
            # Finalize entity if we are at the end of an entity sequence
            if current_entity_tokens:
                entity_phrase = " ".join(current_entity_tokens)
                entity_identifier = _get_or_create_entity_identifier(entity_identifiers, entity_count, current_label, entity_phrase)
                processed_predictions.append(entity_identifier)
                current_entity_tokens.clear()  # Reset for the next entity
                current_label = None

            # Add non-entity word directly to output
            processed_predictions.append(word)
            continue

        entity_index = int(label_key.split("_")[-1])
        label_name = labels_dict.get(entity_index, "O").split("-")[-1]  # Remove B- or I- prefix

        # Check if this label should be processed (masked)
        if label_name != "O" and (tags_to_mask is None or label_name in tags_to_mask):
            if label_name != current_label:
                # Finalize previous entity
                if current_entity_tokens:
                    entity_phrase = " ".join(current_entity_tokens)
                    entity_identifier = _get_or_create_entity_identifier(entity_identifiers, entity_count, current_label, entity_phrase)
                    processed_predictions.append(entity_identifier)
                    current_entity_tokens.clear()

                current_label = label_name  # Update current label

            # Accumulate token for the current entity
            current_entity_tokens.append(word)
        else:
            # Handle non-entity or ignored label
            if current_entity_tokens:
                entity_phrase = " ".join(current_entity_tokens)
                entity_identifier = _get_or_create_entity_identifier(entity_identifiers, entity_count, current_label, entity_phrase)
                processed_predictions.append(entity_identifier)
                current_entity_tokens.clear()
                current_label = None

            processed_predictions.append(word)

    # Process any remaining entity tokens
    if current_entity_tokens:
        entity_phrase = " ".join(current_entity_tokens)
        entity_identifier = _get_or_create_entity_identifier(entity_identifiers, entity_count, current_label, entity_phrase)
        processed_predictions.append(entity_identifier)

    # Construct the marked sentence
    marked_sentence = " ".join(processed_predictions)
    
    return marked_sentence, entity_identifiers

# Helper function to get or create an entity identifier
def _get_or_create_entity_identifier(entity_identifiers, entity_count, label, phrase):
    """
    Retrieve or create a unique identifier for a detected entity based on label and phrase.
    """
    for identifier, stored_phrase in entity_identifiers.items():
        if stored_phrase == phrase:
            return identifier
    # Create a new identifier
    entity_count[label] += 1
    identifier = f"{label.capitalize()}{entity_count[label]}"
    entity_identifiers[identifier] = phrase
    return identifier

# Encrypt each unique entity and create an encrypted dictionary
def encrypt_entities(entity_identifiers, password):
    """Encrypt each unique entity and return a dictionary of encrypted entities."""
    encrypted_entities = {}
    for identifier, phrase in entity_identifiers.items():
        ciphertext, salt, iv = password_encrypt(password, phrase)
        encrypted_entities[identifier] = {
            "ciphertext": ciphertext,
            "salt": salt,
            "iv": iv,
            "original_text": phrase
        }
    return encrypted_entities

# Display only the relevant outputs
def display_results(text, masked_sentence, encrypted_entities):
    """Display original text, masked sentence, and encrypted entities."""
    print("Original Sentence:", text)
    print("Masked Sentence:", masked_sentence)
    print("Encrypted Entities:", json.dumps(encrypted_entities, indent=4, ensure_ascii=False))

print('next')

# Установите пароль для шифрования и расшифровки
password = "mySecretPassword"

# Путь к модели и токен доступа
# Отправляйте другим с осторожностью!
hf_token = "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" 

model_repo = "povchingiz/npk_ner_v06" 
labels_filename = "labels_dict.json"
login(token=hf_token)

# Вход на хагинфейс
hf_login(hf_token)

# Подгрузка модели и списка сущностей
labels_dict = load_labels_dict(model_repo, labels_filename, hf_token)

# Создание пайплайна по модели
ner_pipeline = setup_ner_pipeline(model_repo, hf_token)

print("="*50, "\n", "Этот код выполнился")
password = "mySecretPassword"

import streamlit as st
import json
from transformers import pipeline
from collections import defaultdict

# Streamlit Interface
st.title("Named Entity Recognition App")
st.write("This app identifies entities in text, masks them, and provides encrypted output.")

# Input Text Area
st.write("### Step 1: Enter Text for Analysis")
input_text = st.text_area("Enter your text below:", height=200)

if st.button("Analyze"):
    if input_text.strip():
        # Apply NER model
        masked_sentence, entity_identifiers = apply_model_and_encrypt(input_text, ner_pipeline, labels_dict, password)
        # Display Results
        st.write("### Results")
        st.write("#### Original Text")
        st.text(input_text)
        
        st.write("#### Masked Text")
        st.text(masked_sentence)
        
        st.write("#### Encrypted Entities")
        st.json(entity_identifiers)
    else:
        st.warning("Please enter some text to analyze!")
