import os
import gradio as gr
from document_processor import DocumentProcessor, client, openai_client, collection
import weaviate.classes as wvc
import json
import shutil
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import logging
import uuid
from conversation_store import ConversationStore
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='🔍 %(asctime)s [DOCUMENT-MANAGER] %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger("document_manager")
logger.setLevel(logging.INFO)

# Suppress unnecessary logs
logging.getLogger("gradio").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("fsevents").setLevel(logging.WARNING)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

processor = DocumentProcessor()
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

conversation_store = ConversationStore()

def save_uploaded_file(file, folder="documents"):
    """Save uploaded file to documents folder and return the new path."""
    # Create documents directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    # Get original filename
    filename = os.path.basename(file.name)
    
    # Create new file path
    new_path = os.path.join(folder, filename)
    
    # Copy file to documents folder
    shutil.copy2(file.name, new_path)
    
    return new_path

def check_existing_document(filename):
    """Check if a document with the given filename exists in Weaviate."""
    try:
        # Query Weaviate for documents with matching filename
        results = collection.query.fetch_objects(
            filters=wvc.query.Filter.by_property("filename").equal(filename),
            limit=1  # We only need one result
        )

        # Check if we got any results
        if results and hasattr(results, 'objects') and len(results.objects) > 0:
            # Return the first matching document's ID
            return results.objects[0].properties.get("documentId")
        return None
    except Exception as e:
        logger.error(f"❌ Error checking for existing document: {str(e)}")
        return None

def process_files(files):
    """Process multiple files uploaded through Gradio."""
    results = []
    for file in files:
        try:
            filename = os.path.basename(file.name)
            logger.info(f"📄 Processing file: {filename}")
            
            # Check if file already exists in Weaviate
            existing_doc_id = check_existing_document(filename)
            if existing_doc_id:
                logger.info(f"🔄 Found existing document with filename {filename}, ID: {existing_doc_id}")
                results.append(f"Replacing existing file {filename} (ID: {existing_doc_id})")
            
            # Save file to documents folder
            doc_path = save_uploaded_file(file)
            
            # Process the saved file, passing the existing ID if found
            doc_id = processor.process_document(doc_path, document_id=existing_doc_id)
            results.append(f"Successfully processed {os.path.basename(doc_path)} with ID: {doc_id}")
            
        except Exception as e:
            logger.error(f"❌ Error processing {os.path.basename(file.name)}: {str(e)}")
            results.append(f"Error processing {os.path.basename(file.name)}: {str(e)}")
    return "\n".join(results)

def process_folder(folder_files):
    """Process all files from an uploaded folder."""
    if not folder_files:
        return "No files uploaded"
        
    results = []
    documents_dir = "documents"
    os.makedirs(documents_dir, exist_ok=True)
    
    for file in folder_files:
        try:
            filename = os.path.basename(file.name)
            logger.info(f"📄 Processing folder file: {filename}")
            
            # Check if file already exists in Weaviate
            existing_doc_id = check_existing_document(filename)
            if existing_doc_id:
                logger.info(f"🔄 Found existing document with filename {filename}, ID: {existing_doc_id}")
                results.append(f"Replacing existing file {filename} (ID: {existing_doc_id})")
            
            # Create destination path in documents folder
            dst_path = os.path.join(documents_dir, filename)
            
            # Copy file to documents folder
            shutil.copy2(file.name, dst_path)
            
            # Process the saved file, passing the existing ID if found
            doc_id = processor.process_document(dst_path, document_id=existing_doc_id)
            results.append(f"Successfully processed {filename} with ID: {doc_id}")
            
        except Exception as e:
            logger.error(f"❌ Error processing {filename}: {str(e)}")
            results.append(f"Error processing {filename}: {str(e)}")
    
    return "\n".join(results)

def query_documents(query: str, doc_id: str = None, limit: int = 5, context: str = ""):
    """Query documents using semantic search and generate LLM response."""
    try:
        # Check if it's a greeting
        greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
        if query.lower().strip() in greetings:
            return "👋 Hello! How can I help you today?"
            
        # Get query embedding
        response = openai_client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding

        # Build query filter for multiple document IDs
        query_filter = None
        if doc_id:
            # Split and clean document IDs
            doc_ids = [id.strip() for id in doc_id.split(',') if id.strip()]
            if doc_ids:
                # Create OR filter for multiple document IDs
                query_filter = wvc.query.Filter.by_property("documentId").contains_any(doc_ids)

        # Request more results initially to account for duplicates
        search_limit = limit * 2  # Request more results initially

        # Execute hybrid search with both semantic and vector search
        results = collection.query.hybrid(
            query=query,  # for text search
            vector=query_embedding,  # for vector search
            alpha=0.5,  # balance between keyword (0) and vector (1) search
            filters=query_filter,
            limit=search_limit,
            return_properties=["text", "documentId", "filename", "chunkIndex", "metadata"],
            return_metadata=wvc.query.MetadataQuery(
                score=True,  # Overall hybrid search score
            )
        ).objects

        # Format results
        if not results:
            return "📝 Answer:\nI don't have any information about that topic in the current documents."

        # Deduplicate results based on text content
        unique_results = []
        seen_texts = set()
        
        for obj in results:
            if len(unique_results) >= limit:  # Stop if we have enough unique results
                break
                
            text = obj.properties['text'].strip()
            if text not in seen_texts:
                seen_texts.add(text)
                unique_results.append(obj)

        # Prepare context for LLM with better structure
        contexts = []
        for obj in unique_results:
            props = obj.properties
            score = obj.metadata.score if hasattr(obj.metadata, 'score') else "N/A"
            
            try:
                metadata = json.loads(props['metadata']) if props.get('metadata') else {}
            except json.JSONDecodeError:
                metadata = {}
            
            # Format metadata information
            meta_info = []
            if metadata.get('source_type') == 'PDF':
                # Handle both single page number and page numbers list
                page_numbers = metadata.get('page_numbers', [])
                if not page_numbers and metadata.get('page_number'):
                    page_numbers = [metadata['page_number']]
                if page_numbers:
                    meta_info.append(f"Page(s): {', '.join(map(str, sorted(page_numbers)))}")
                
                # Add element types
                element_types = metadata.get('element_types', [])
                if not element_types and metadata.get('element_type'):
                    element_types = [metadata['element_type']]
                if element_types:
                    meta_info.append(f"Type(s): {', '.join(element_types)}")
                
                # Add headings if available
                if metadata.get('headings'):
                    meta_info.append(f"Heading: {metadata['headings']}")
            
            # Format each chunk with its metadata and scores
            chunk_context = f"""
[Document: {props['filename']}
{' | '.join(meta_info) if meta_info else ''}
Chunk: {props['chunkIndex']}
Score: {score}]

{props['text']}
"""
            contexts.append(chunk_context)

        # Generate LLM response with structured context
        llm_prompt = f"""Based on the following context chunks from a document, answer the question: "{query}"

Context:
{'-' * 80}
{context}
{'-' * 80}

{''.join(contexts)}

Important: If user greets you, greet the user and ask how you can help them. Don't tell i dont have context to answer the greeting. Only provide information that is directly supported by the given context. If the context doesn't contain enough information to fully answer the question, acknowledge this limitation in your response."""

        llm_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides accurate information based strictly on the given context. If the context doesn't fully answer the question, acknowledge this limitation."},
                {"role": "user", "content": llm_prompt}
            ],
            temperature=0
        )

        # Format final output with LLM response and source information
        output = ["📝 Answer:\n" + llm_response.choices[0].message.content + "\n"]
        output.append("\n📚 Sources:")
        
        for i, obj in enumerate(unique_results):
            props = obj.properties
            score = obj.metadata.score if hasattr(obj.metadata, 'score') else "N/A"
            
            try:
                metadata = json.loads(props['metadata']) if props.get('metadata') else {}
            except json.JSONDecodeError:
                metadata = {}
            
            # Build source information with all relevant metadata
            source_info = [
                f"\n{i+1}. Document: {props['filename']}",
                f"   ID: {props['documentId']}"
            ]
            
            # Add metadata information
            if metadata.get('source_type') == 'PDF':
                # Handle both single page number and page numbers list
                page_numbers = metadata.get('page_numbers', [])
                if not page_numbers and metadata.get('page_number'):
                    page_numbers = [metadata['page_number']]
                if page_numbers:
                    source_info.append(f"   Page(s): {', '.join(map(str, sorted(page_numbers)))}")
                
                # Add headings if available
                if metadata.get('headings'):
                    source_info.append(f"   Heading: {metadata['headings']}")
            
            source_info.extend([
                f"   Chunk: {props['chunkIndex']}",
                f"   Score: {score}"
            ])
            
            output.append('\n'.join(source_info))

        return "\n".join(output)

    except Exception as e:
        return f"Error during query: {str(e)}"

def get_ingested_documents():
    """Get list of all ingested documents with their IDs."""
    try:
        # Create a dictionary to store unique documents
        unique_docs = {}
        
        # Use iterator to get all documents
        for obj in collection.iterator():
            documentId = obj.properties["documentId"]
            filename = obj.properties["filename"]
            
            unique_docs[documentId] = {
                "filename": filename,
                "documentId": documentId,
                "weaviate_id": obj.uuid
            }
        
        # Convert dictionary values to list
        documents = list(unique_docs.values())
        return documents
    except Exception as e:
        print(f"Error fetching documents: {str(e)}")
        return []

def delete_document(filename: str, documentId: str):
    """Delete document from both documents folder and Weaviate."""
    try:
        # Delete from documents folder
        file_path = os.path.join("documents", filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Delete from Weaviate using documentId property
        collection.data.delete_many(
            where=wvc.query.Filter.by_property("documentId").equal(documentId)
        )
        
        return f"Successfully deleted {filename} (ID: {documentId})"
    except Exception as e:
        return f"Error deleting {filename}: {str(e)}"

def update_selection_info(table_data):
    """Update info about selected documents."""
    logger.info("🔄 Selection update triggered")
    
    # Handle None case
    if table_data is None:
        logger.warning("⚠️ Received None table data")
        return "No documents selected"
    
    try:
        # Convert DataFrame to list if needed
        if hasattr(table_data, 'values'):
            table_data = table_data.values.tolist()
            logger.info("📊 Converted DataFrame to list")
        
        # Handle empty or invalid table data
        if not isinstance(table_data, list) or len(table_data) == 0:
            logger.warning(f"⚠️ Invalid table data: {type(table_data)}")
            return "No documents selected"
        
        logger.info(f"📊 Processing {len(table_data)} rows")
        logger.info(f"📋 Raw table data: {table_data}")
        
        selected_docs = []
        for i, row in enumerate(table_data):
            try:
                # Validate row data
                if not isinstance(row, (list, tuple)) or len(row) < 2:
                    logger.warning(f"⚠️ Invalid row format at index {i}: {row}")
                    continue
                
                # Extract and validate row data
                filename = str(row[0]) if row[0] is not None else "Unknown"
                doc_id = str(row[1]) if row[1] is not None else "Unknown"
                
                # Log row status
                status = "✅ Selected"
                logger.info(f"Row {i}: {status} | {filename} (ID: {doc_id})")
                
                # Add to selected documents if selected
                doc_info = f"{filename} (ID: {doc_id})"
                selected_docs.append(f"Selected: {doc_info}")
                
            except Exception as e:
                logger.error(f"❌ Error processing row {i}: {str(e)}")
                logger.error(f"Row data: {row}")
        
        # Return results
        if not selected_docs:
            logger.info("ℹ️ No documents currently selected")
            return "No documents selected"
        
        result = "\n".join(selected_docs)
        logger.info(f"📋 Selection summary: {len(selected_docs)} documents selected")
        logger.info(f"📝 Selected documents:\n{result}")
        return result
    except Exception as e:
        logger.error(f"❌ Error in update_selection_info: {str(e)}")
        logger.error(f"Stack trace: {e.__traceback__}")
        return "Error processing selection"

def handle_delete_selected(doc_ids: list):
    """Handle deletion of selected documents."""
    logger.info("🗑️ Delete selected triggered")
    
    if not doc_ids:
        logger.warning("❌ No document IDs received")
        return "No documents selected for deletion", [], "No documents selected", []
    
    try:
        results = []
        for doc_id in doc_ids:
            # Get document info first
            docs = get_ingested_documents()
            doc = next((d for d in docs if d["documentId"] == doc_id), None)
            
            if not doc:
                logger.warning(f"⚠️ Document with ID {doc_id} not found")
                results.append(f"Document with ID {doc_id} not found")
                continue
                
            filename = doc["filename"]
            logger.info(f"🗑️ Preparing to delete: {filename} (ID: {doc_id})")
            
            # Delete from Weaviate database
            collection.data.delete_many(
                where=wvc.query.Filter.by_property("documentId").equal(doc_id)
            )
            logger.info("✅ Successfully deleted document from database")
            
            # Delete physical file
            file_path = os.path.join("documents", filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                results.append(f"Successfully deleted {filename} (ID: {doc_id})")
            else:
                results.append(f"Warning: File {filename} not found in documents folder")
        
        # Get fresh data after deletion
        updated_data = refresh_gradio_state()
        result_message = "\n".join(results) if results else "No documents were deleted"
        return result_message, updated_data, "No documents selected", []
    except Exception as e:
        logger.error(f"❌ Error in handle_delete_selected: {str(e)}")
        return f"Error deleting documents: {str(e)}", [], "Error occurred", []

def delete_all_documents():
    """Delete all documents from both documents folder and Weaviate."""
    try:
        # Get all documents first
        documents = get_ingested_documents()
        if not documents:
            return "No documents found to delete"
            
        results = []
        doc_ids = [doc["documentId"] for doc in documents]
            
        # First delete from Weaviate using documentId property
        try:
            collection.data.delete_many(
                where=wvc.query.Filter.by_property("documentId").contains_any(doc_ids)
            )
        except Exception as e:
            return f"Error deleting from Weaviate: {str(e)}"
            
        # Then delete files from documents folder
        for doc in documents:
            try:
                file_path = os.path.join("documents", doc["filename"])
                if os.path.exists(file_path):
                    os.remove(file_path)
                    results.append(f"Successfully deleted {doc['filename']} (ID: {doc['documentId']})")
                else:
                    results.append(f"Warning: File {doc['filename']} not found in documents folder")
            except Exception as e:
                results.append(f"Error deleting file {doc['filename']}: {str(e)}")
            
        return f"Deletion results:\n" + "\n".join(results)
    except Exception as e:
        return f"Error deleting all documents: {str(e)}"

def update_document_list():
    """Update the document list table."""
    try:
        # Get list of ingested documents
        docs = get_ingested_documents()
        logger.info(f"📋 Found {len(docs)} documents in database")
        
        # Format data for table
        table_data = []
        for doc in docs:
            doc_id = doc.get('documentId')
            filename = doc.get('filename', 'Unknown')
            table_data.append([filename, doc_id])
            
        if not table_data:
            logger.info("📭 No documents found in database")
            return [], "No documents found."
            
        logger.info("✅ Document list updated successfully")
        return table_data, "Document list refreshed successfully."
        
    except Exception as e:
        error_msg = f"Error updating document list: {str(e)}"
        logger.error(f"❌ {error_msg}")
        return [], error_msg

def handle_delete_all():
    """Handle delete all documents."""
    try:
        result = delete_all_documents()
        updated_data, _ = update_document_list()
        return updated_data, result
    except Exception as e:
        print(f"Error in handle_delete_all: {str(e)}")
        return [], f"Error deleting all documents: {str(e)}"

def handle_row_select(table_data, evt: gr.SelectData, selected_doc_ids: list = None):
    """Handle row selection in the documents table."""
    try:
        if evt is None:
            return "No document selected", []
        
        logger.info(f"Selection event: index={evt.index}, value={evt.value}, data type={type(table_data)}")
        
        # Initialize current selections
        current_selections = selected_doc_ids if selected_doc_ids is not None else []
        
        # Handle pandas DataFrame
        if isinstance(table_data, pd.DataFrame):
            # Get the row index (first element of the index tuple)
            row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            
            if 0 <= row_idx < len(table_data):
                # Get row data using integer index
                row = table_data.iloc[row_idx]
                # Get values by column position since we know the structure
                filename = str(row.iloc[0]) if pd.notna(row.iloc[0]) else "Unknown"
                doc_id = str(row.iloc[1]) if pd.notna(row.iloc[1]) else "Unknown"
                
                logger.info(f"Selected row {row_idx}: filename={filename}, doc_id={doc_id}")
                
                # Toggle selection
                if doc_id in current_selections:
                    current_selections.remove(doc_id)
                    logger.info(f"Deselected document: {filename} (ID: {doc_id})")
                else:
                    current_selections.append(doc_id)
                    logger.info(f"Added document to selection: {filename} (ID: {doc_id})")
                
                # Format selection info text
                if not current_selections:
                    return "No documents selected", []
                
                # Get info for all selected documents
                selected_docs_info = []
                for selected_id in current_selections:
                    selected_row = table_data[table_data.iloc[:, 1] == selected_id]
                    if not selected_row.empty:
                        selected_filename = selected_row.iloc[0, 0]
                        selected_docs_info.append(f"Selected: {selected_filename} (ID: {selected_id})")
                
                selection_text = "\n".join(selected_docs_info)
                return selection_text, current_selections
            else:
                logger.warning(f"Invalid row index: {row_idx}")
                return "Invalid selection", current_selections
        else:
            logger.warning(f"Unexpected data type: {type(table_data)}")
            return "Invalid data format", current_selections
        
    except Exception as e:
        logger.error(f"Error in selection handler: {str(e)}")
        logger.error(f"Selection event data: {evt}")
        return "Error processing selection", current_selections

def start_new_chat():
    """Start a new chat conversation."""
    try:
        # Generate new conversation ID
        new_conv_id = str(uuid.uuid4())
        
        # Create conversation in store
        if not conversation_store.create_conversation(new_conv_id):
            return None, [], "Error starting new chat"
            
        # Add system message
        conversation_store.add_message(
            new_conv_id,
            "system",
            "I am a helpful assistant that provides accurate information based on the documents. Be conversational and engaging while maintaining accuracy."
        )
        
        return new_conv_id, [], None
    except Exception as e:
        logger.error(f"Error starting new chat: {str(e)}")
        return None, [], f"Error starting new chat: {str(e)}"

def chat(message, conv_id, history):
    """Handle chat messages."""
    try:
        if not message:
            return conv_id, history, ""
            
        if not conv_id:
            # Start new conversation if none exists
            conv_id = str(uuid.uuid4())
            conversation_store.create_conversation(conv_id)
            conversation_store.add_message(
                conv_id,
                "system",
                "I am a helpful assistant that provides accurate information based on the documents. Be conversational and engaging while maintaining accuracy."
            )
        
        # Add user message to history
        conversation_store.add_message(conv_id, "user", message)
        
        # Get last 10 conversation pairs for context (20 messages)
        chat_history = conversation_store.get_conversation_history(conv_id, limit=10)
        
        # Format conversation history for context (excluding sources)
        conversation_context = "\n".join([
            f"{msg['role']}: {msg['content'].split('📚 Sources:')[0].strip()}"  # Only take content before sources
            for msg in chat_history  # No need to reverse, keep chronological order
        ])

        # Check if message is a simple acknowledgment or greeting
        simple_responses = ["okay", "ok", "fine", "alright", "thanks", "thank you", "got it", "understood", "i see"]
        greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
        
        message_lower = message.lower().strip()
        is_simple_response = message_lower in simple_responses
        is_greeting = message_lower in greetings
        
        if is_simple_response or is_greeting:
            # Skip document search for simple responses and greetings
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Respond naturally to greetings and acknowledgments."},
                {"role": "user", "content": f"""Based on this conversation history, respond to: "{message}"

Conversation History:
{conversation_context}

Important: Respond naturally to the user's {['acknowledgment', 'greeting'][is_greeting]}."""}
            ]
            
            llm_response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0
            )
            response = llm_response.choices[0].message.content
            
        else:
            try:
                # Get query embedding for actual queries
                response = openai_client.embeddings.create(
                    input=message,
                    model="text-embedding-3-small"
                )
                query_embedding = response.data[0].embedding

                # Search in Weaviate
                results = collection.query.hybrid(
                    query=message,
                    vector=query_embedding,
                    alpha=0.5,
                    limit=5,
                    return_properties=["text", "documentId", "filename", "chunkIndex", "metadata"],
                    return_metadata=wvc.query.MetadataQuery(score=True)
                ).objects

                # Format document context if results found
                if results:
                    doc_contexts = []
                    for obj in results:
                        props = obj.properties
                        score = obj.metadata.score if hasattr(obj.metadata, 'score') else "N/A"
                        
                        chunk_context = f"""
[Document: {props['filename']}
Chunk: {props['chunkIndex']}
Score: {score}]

{props['text']}
"""
                        doc_contexts.append(chunk_context)
                    
                    document_context = "\n".join(doc_contexts)
                else:
                    document_context = ""

                # Generate LLM response
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that provides accurate information based on the given context. If no relevant document context is available, respond based on the conversation history."},
                    {"role": "user", "content": f"""Based on the following context, answer the question: "{message}"

Conversation History:
{conversation_context}

Document Context:
{document_context}

Important: If no document context is available, respond based on the conversation history. If neither context is helpful, acknowledge that you don't have relevant information."""}
                ]

                llm_response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0
                )
                
                response = llm_response.choices[0].message.content

                # Add source information if document context was used
                if document_context:
                    response += "\n\n📚 Sources:\n"
                    for obj in results:
                        props = obj.properties
                        score = obj.metadata.score if hasattr(obj.metadata, 'score') else "N/A"
                        response += f"\nDocument: {props['filename']}\nID: {props['documentId']}\nChunk: {props['chunkIndex']}\nScore: {score}\n"

            except Exception as e:
                logger.error(f"Error in document search: {str(e)}")
                # Fallback to conversation-only response
                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Respond based on the conversation history only."},
                    {"role": "user", "content": f"""Based on this conversation history, respond to: "{message}"

Conversation History:
{conversation_context}

Important: If you can't provide a meaningful response, acknowledge that you don't have enough information."""}
                ]

                llm_response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0
                )
                response = llm_response.choices[0].message.content
        
        # Store only the response without sources in conversation history
        conversation_store.add_message(conv_id, "assistant", response.split("📚 Sources:")[0].strip())
        
        # Update UI history with full response (including sources)
        history = history or []
        history.append((message, response))
        
        return conv_id, history, ""
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        return conv_id, history, f"Error: {str(e)}"

def refresh_gradio_state():
    """Refresh Gradio interface state."""
    try:
        # Get fresh document list
        docs = get_ingested_documents()
        table_data = []
        for doc in docs:
            doc_id = doc.get('documentId')
            filename = doc.get('filename', 'Unknown')
            table_data.append([filename, doc_id])
            
        return table_data
    except Exception as e:
        logger.error(f"Error refreshing Gradio state: {str(e)}")
        return []

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Document Processing System")
    
    with gr.Tab("Upload Files"):
        files_input = gr.File(file_count="multiple", label="Upload Documents")
        files_output = gr.Textbox(label="Processing Results")
        files_button = gr.Button("Process Files")
        files_button.click(process_files, inputs=[files_input], outputs=[files_output])
    
    with gr.Tab("Upload Folder"):
        folder_input = gr.File(file_count="directory", label="Upload Folder")
        folder_output = gr.Textbox(label="Processing Results")
        folder_button = gr.Button("Process Folder")
        folder_button.click(process_folder, inputs=[folder_input], outputs=[folder_output])

    with gr.Tab("Manage Documents", id="manage_docs") as manage_tab:
        with gr.Row():
            refresh_button = gr.Button("🔄 Refresh List")
            delete_selected_btn = gr.Button("🗑️ Delete Selected", variant="secondary")
            delete_all_btn = gr.Button("🗑️ Delete All Documents", variant="stop")
        status_text = gr.Textbox(label="Status", interactive=False)
        selection_info = gr.Textbox(label="Selected Documents", interactive=False)
        
        # Store selected document IDs (multiple)
        selected_doc_ids = gr.State(value=[])
        
        documents_table = gr.Dataframe(
            headers=["Filename", "Document ID"],
            datatype=["str", "str"],
            interactive=True,
            value=[],
            row_count=(0, "dynamic"),
            col_count=(2, "fixed"),
            wrap=True
        )

        # Set up event handlers
        refresh_button.click(
            fn=update_document_list,
            outputs=[documents_table, status_text]
        )
        
        # Auto-refresh when tab is selected
        manage_tab.select(
            fn=update_document_list,
            outputs=[documents_table, status_text]
        )
        
        documents_table.select(
            fn=handle_row_select,
            inputs=[documents_table, selected_doc_ids],
            outputs=[selection_info, selected_doc_ids]
        )
        
        delete_selected_btn.click(
            fn=handle_delete_selected,
            inputs=[selected_doc_ids],
            outputs=[status_text, documents_table, selection_info, selected_doc_ids]
        )
        
        delete_all_btn.click(
            fn=handle_delete_all,
            outputs=[documents_table, status_text]
        )

    with gr.Tab("Search Documents"):
        with gr.Row():
            query_input = gr.Textbox(label="Search Query", placeholder="Enter your search query...")
            doc_id_input = gr.Textbox(label="Document IDs (optional, comma-separated)", placeholder="doc1,doc2,...")
            limit_input = gr.Number(label="Number of retrieved chunks", value=5, minimum=1, maximum=20)
        query_output = gr.Textbox(label="Results")
        query_button = gr.Button("Search")
        query_button.click(query_documents, inputs=[query_input, doc_id_input, limit_input], outputs=[query_output])

    with gr.Tab("Chat"):
        conversation_id = gr.State(value=None)
        
        with gr.Row():
            with gr.Column(scale=3):
                chat_history = gr.Chatbot(label="Chat History", height=400)
                chat_input = gr.Textbox(label="Type your message", placeholder="Ask me anything about the documents...")
                with gr.Row():
                    chat_button = gr.Button("Send", variant="primary")
                    clear_button = gr.Button("New Chat", variant="secondary")
        
        chat_button.click(
            chat,
            inputs=[chat_input, conversation_id, chat_history],
            outputs=[conversation_id, chat_history, chat_input]
        )
        
        clear_button.click(
            start_new_chat,
            outputs=[conversation_id, chat_history, chat_input]
        )
        
        # Initialize chat on load
        demo.load(
            start_new_chat,
            outputs=[conversation_id, chat_history, chat_input]
        )

# Mount Gradio app
app = gr.mount_gradio_app(app, demo, path="/ui")  # Mount at /ui instead of /

# FastAPI endpoints
@app.get("/api/documents")
async def get_documents():
    """API endpoint to get all documents."""
    documents = get_ingested_documents()
    return {"documents": documents}

@app.post("/api/ingest/files")
async def ingest_files(request: Request):
    """API endpoint to ingest multiple files."""
    try:
        form = await request.form()
        files = form.getlist("files")
        if not files:
            return {"status": "error", "message": "No files provided"}
        
        results = []
        for file in files:
            try:
                filename = file.filename
                logger.info(f"📄 Processing file: {filename}")
                
                # Check if file already exists in Weaviate
                existing_doc_id = check_existing_document(filename)
                if existing_doc_id:
                    logger.info(f"🔄 Found existing document with filename {filename}, ID: {existing_doc_id}")
                    results.append({
                        "status": "info",
                        "filename": filename,
                        "documentId": existing_doc_id,
                        "message": f"Replacing existing file {filename} (ID: {existing_doc_id})"
                    })
                
                # Save file directly to documents folder
                os.makedirs("documents", exist_ok=True)
                file_path = os.path.join("documents", filename)
                
                # Save uploaded file content
                contents = await file.read()
                with open(file_path, "wb") as f:
                    f.write(contents)
                
                # Process the file
                doc_id = processor.process_document(file_path, document_id=existing_doc_id)
                
                results.append({
                    "status": "success",
                    "filename": filename,
                    "documentId": doc_id,
                    "message": f"Successfully ingested {filename} with ID: {doc_id}"
                })
            except Exception as e:
                results.append({
                    "status": "error",
                    "filename": filename,
                    "message": f"Error ingesting {filename}: {str(e)}"
                })
        
        # Get updated document list for Gradio
        table_data = refresh_gradio_state()
        
        # Update Gradio interface using its event system
        try:
            documents_table.update(value=table_data)
        except Exception as e:
            logger.error(f"Failed to update Gradio interface: {str(e)}")
        
        return {
            "status": "completed",
            "results": results
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/ingest/folder")
async def ingest_folder(request: Request):
    """API endpoint to ingest a folder of files."""
    try:
        form = await request.form()
        files = form.getlist("files")
        if not files:
            return {"status": "error", "message": "No files provided"}
        
        results = []
        for file in files:
            try:
                filename = file.filename
                logger.info(f"📄 Processing folder file: {filename}")
                
                # Check if file already exists in Weaviate
                existing_doc_id = check_existing_document(filename)
                if existing_doc_id:
                    logger.info(f"🔄 Found existing document with filename {filename}, ID: {existing_doc_id}")
                    results.append({
                        "status": "info",
                        "filename": filename,
                        "documentId": existing_doc_id,
                        "message": f"Replacing existing file {filename} (ID: {existing_doc_id})"
                    })
                
                # Save file directly to documents folder
                os.makedirs("documents", exist_ok=True)
                file_path = os.path.join("documents", filename)
                
                # Save uploaded file content
                contents = await file.read()
                with open(file_path, "wb") as f:
                    f.write(contents)
                
                # Process the file with existing document ID if found
                doc_id = processor.process_document(file_path, document_id=existing_doc_id)
                
                results.append({
                    "status": "success",
                    "filename": filename,
                    "documentId": doc_id,
                    "message": f"Successfully ingested {filename} with ID: {doc_id}"
                })
            except Exception as e:
                results.append({
                    "status": "error",
                    "filename": filename,
                    "message": f"Error ingesting {filename}: {str(e)}"
                })
        
        # Get updated document list for Gradio
        table_data = refresh_gradio_state()
        
        # Update Gradio interface using its event system
        try:
            documents_table.update(value=table_data)
        except Exception as e:
            logger.error(f"Failed to update Gradio interface: {str(e)}")
        
        return {
            "status": "completed",
            "results": results
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.delete("/api/del_docs")
async def delete_documents(request: Request):
    """API endpoint to delete multiple documents by their IDs."""
    try:
        data = await request.json()
        doc_ids = data.get("documentIds", [])
        if not doc_ids:
            return {"status": "error", "message": "No document IDs provided"}
        
        results = []
        for doc_id in doc_ids:
            try:
                # Get document info first
                docs = get_ingested_documents()
                doc = next((d for d in docs if d["documentId"] == doc_id), None)
                if doc:
                    result = delete_document(doc["filename"], doc_id)
                    results.append({
                        "status": "success",
                        "documentId": doc_id,
                        "message": result
                    })
                else:
                    results.append({
                        "status": "error",
                        "documentId": doc_id,
                        "message": f"Document with ID {doc_id} not found"
                    })
            except Exception as e:
                results.append({
                    "status": "error",
                    "documentId": doc_id,
                    "message": f"Error deleting document {doc_id}: {str(e)}"
                })
        
        # Get updated document list for Gradio
        table_data = refresh_gradio_state()
        
        # Update Gradio interface using its event system
        try:
            documents_table.update(value=table_data)
        except Exception as e:
            logger.error(f"Failed to update Gradio interface: {str(e)}")
        
        return {
            "status": "completed",
            "results": results
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.delete("/api/documents/all")
async def delete_all():
    """API endpoint to delete all documents."""
    try:
        result = delete_all_documents()
        return {
            "status": "success",
            "message": result
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/search")
async def search(request: Request):
    """Search documents with a query."""
    try:
        data = await request.json()
        query = data.get("query")
        doc_ids = data.get("documentIds", [])  # Optional array of document IDs
        
        if not query:
            return {"status": "error", "message": "Query is required"}
            
        # Convert doc_ids list to comma-separated string for query_documents
        doc_ids_str = ",".join(doc_ids) if doc_ids else None
        response = query_documents(query, doc_id=doc_ids_str)
        
        # Parse the response to separate answer and sources
        parts = response.split("\n📚 Sources:")
        answer = parts[0].replace("📝 Answer:\n", "").strip()
        
        # Parse and structure the sources as JSON
        sources = []
        if len(parts) > 1:
            # Split source text into individual source entries
            source_entries = parts[1].strip().split("\n\n")
            for entry in source_entries:
                if not entry.strip():
                    continue
                    
                # Parse each source entry
                source_dict = {}
                lines = entry.strip().split("\n")
                for line in lines:
                    line = line.strip()
                    # Handle numbered document entries (e.g., "1. Document: file.pdf")
                    if "Document:" in line:
                        source_dict["filename"] = line.split("Document:")[1].strip()
                    elif "ID:" in line:
                        source_dict["documentId"] = line.split("ID:")[1].strip()
                    elif "Page(s):" in line:
                        pages_str = line.split("Page(s):")[1].strip()
                        source_dict["pages"] = [int(p.strip()) for p in pages_str.split(",")]
                    elif "Heading:" in line:
                        source_dict["heading"] = line.split("Heading:")[1].strip()
                    elif "Chunk:" in line:
                        source_dict["chunkIndex"] = int(line.split("Chunk:")[1].strip())
                    elif "Score:" in line:
                        source_dict["score"] = float(line.split("Score:")[1].strip())
                
                if source_dict:  # Only add if we parsed some data
                    sources.append(source_dict)
        
        return {
            "status": "success",
            "answer": answer,
            "sources": sources
        }
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/api/chat/start")
async def start_conversation(request: Request):
    """Start a new chat conversation."""
    try:
        data = await request.json()
        query = data.get("query")
        doc_ids = data.get("documentIds", [])  # Optional array of document IDs
        
        if not query:
            return {"status": "error", "message": "Query is required"}
            
        # Create new conversation
        conversation_id = str(uuid.uuid4())
        if not conversation_store.create_conversation(conversation_id):
            return {"status": "error", "message": "Failed to create conversation"}
            
        # Add system message
        conversation_store.add_message(
            conversation_id,
            "system",
            "I am a helpful assistant that can answer questions based on the ingested documents."
        )
        
        # Store document IDs in a special system message with a prefix
        if doc_ids:
            conversation_store.add_message(
                conversation_id,
                "system",
                f"__DOCUMENT_IDS__:{json.dumps(doc_ids)}"
            )
        
        # Add user query
        conversation_store.add_message(conversation_id, "user", query)
        
        # Get response
        doc_ids_str = ",".join(doc_ids) if doc_ids else None
        response = query_documents(query, doc_id=doc_ids_str)
        
        # Add assistant response
        conversation_store.add_message(conversation_id, "assistant", response)
        
        # Parse the response to get just the answer without sources
        answer = response.split("\n📚 Sources:")[0].replace("📝 Answer:\n", "").strip()
        
        return {
            "status": "success",
            "conversationId": conversation_id,
            "answer": answer
        }
    except Exception as e:
        logger.error(f"Error starting conversation: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/api/chat/continue")
async def continue_conversation(request: Request):
    """Continue an existing chat conversation."""
    try:
        data = await request.json()
        conversation_id = data.get("conversationId")
        query = data.get("query")
        
        if not conversation_id or not query:
            return {"status": "error", "message": "Missing conversationId or query"}
        
        # Check if conversation exists
        if not conversation_store.conversation_exists(conversation_id):
            return {"status": "error", "message": "Conversation not found"}
        
        # Get conversation history - already in chronological order from oldest to newest
        chat_history = conversation_store.get_conversation_history(conversation_id)
        
        # Get all user messages except system messages
        user_messages = [msg for msg in chat_history if msg['role'] == 'user']
        
        # Create a system prompt that includes instructions for handling meta-questions
        system_prompt = """You are a helpful assistant that provides accurate information based on the given context. 
If the user asks about previous questions or conversation history (e.g., "what did I ask?", "show my questions", etc.), 
list all their previous questions in chronological order, numbered from 1.

For such meta-questions about conversation history, DO NOT include the current question in the list.
Format each question with quotes, like: 1. "actual question here"

For all other questions, provide information based on the document context. If no relevant document context is available, 
respond based on the conversation history. If neither context is helpful, acknowledge this limitation."""

        # Check if the query is about conversation history using GPT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Determine if this is a meta-question about conversation history: "{query}"
If it is, respond with the previous questions from this conversation history:

User's previous questions:
{json.dumps([msg['content'] for msg in user_messages if msg['content'] != query])}"""}
        ]

        meta_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0
        )
        
        response_text = meta_response.choices[0].message.content
        
        # If GPT formatted a list of questions, it means it identified this as a meta-question
        # Check if the response contains numbered questions (e.g., "1.", "2.", etc.) and quotes
        has_numbered_questions = any(str(i)+"." in response_text for i in range(1, 10))
        if has_numbered_questions and '"' in response_text:
            # Add the current query to history
            conversation_store.add_message(conversation_id, "user", query)
            conversation_store.add_message(conversation_id, "assistant", response_text)
            
            return {
                "status": "success",
                "answer": response_text
            }
        
        # Get document IDs from special system message
        doc_ids = []
        for msg in chat_history:
            if msg['role'] == 'system' and msg['content'].startswith('__DOCUMENT_IDS__:'):
                try:
                    doc_ids = json.loads(msg['content'].replace('__DOCUMENT_IDS__:', ''))
                    break
                except json.JSONDecodeError:
                    continue
        
        # Format conversation history as sequential message pairs
        conversation_context = ""
        for i in range(0, len(chat_history)-1, 2):  # Step by 2 to get pairs
            if chat_history[i]['role'] == 'user' and i+1 < len(chat_history) and chat_history[i+1]['role'] == 'assistant':
                pair_num = (i // 2) + 1
                user_msg = chat_history[i]['content'].split('📚 Sources:')[0].strip()
                assistant_msg = chat_history[i+1]['content'].split('📚 Sources:')[0].strip()
                conversation_context += f"\nmessage pair {pair_num}\nuser: {user_msg}\nassistant: {assistant_msg}\n"
        
        # Add the current query as the next potential pair
        next_pair_num = (len(chat_history) // 2) + 1
        conversation_context += f"\nand now my new query - this will become message pair {next_pair_num} eventually\nuser: {query}\n"
        
        # Process normal query with context from history, using the same document IDs
        doc_ids_str = ",".join(doc_ids) if doc_ids else None
        response = query_documents(query, doc_id=doc_ids_str, context=conversation_context)
        
        # Store only the answer part in conversation history
        answer = response.split("\n📚 Sources:")[0].replace("📝 Answer:\n", "").strip()
        conversation_store.add_message(conversation_id, "user", query)
        conversation_store.add_message(conversation_id, "assistant", answer)
        
        return {
            "status": "success",
            "answer": answer
        }
    except Exception as e:
        logger.error(f"Error continuing conversation: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/api/chat/{conversation_id}/history")
async def get_chat_history(conversation_id: str):
    """Get the history of a specific conversation."""
    try:
        if not conversation_store.conversation_exists(conversation_id):
            return {"status": "error", "message": "Conversation not found"}
            
        history = conversation_store.get_conversation_history(conversation_id)
        return {
            "status": "success",
            "conversationId": conversation_id,
            "history": history
        }
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.delete("/api/chat/{conversation_id}")
async def delete_chat(conversation_id: str):
    """Delete a specific conversation."""
    try:
        if not conversation_store.conversation_exists(conversation_id):
            return {"status": "error", "message": "Conversation not found"}
            
        if conversation_store.delete_conversation(conversation_id):
            return {
                "status": "success",
                "message": f"Conversation {conversation_id} deleted"
            }
        else:
            return {"status": "error", "message": "Failed to delete conversation"}
    except Exception as e:
        logger.error(f"Error deleting chat: {str(e)}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(app, host="0.0.0.0", port=7860)
    finally:
        client.close()  # Ensure the client connection is closed when the app exits 