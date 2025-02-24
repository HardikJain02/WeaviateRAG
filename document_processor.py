import os
import uuid
import json
from typing import Optional, Dict, Any, List
import torch
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker
from docling.datamodel.document import ConversionResult, DoclingDocument
from docling.datamodel.base_models import TextElement
from docling_core.types.doc import TextItem, NodeItem, DocItemLabel
import weaviate
import weaviate.classes as wvc
from dotenv import load_dotenv
from openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configure device for Docling
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS GPU")
else:
    device = torch.device("cpu")
    print("Warning: No GPU available. Using CPU for document processing.")

# Configure Weaviate connection with longer timeout and init config
try:
    additional_config = wvc.init.AdditionalConfig(
        timeout=wvc.init.Timeout(init=5)  # 5 seconds timeout
    )
    
    client = weaviate.connect_to_local(
        additional_config=additional_config,
        skip_init_checks=True  # Skip gRPC checks if they fail
    )
    
    # Create or get the collection
    if not client.collections.exists("Document"):
        collection = client.collections.create(
            name="Document",
            vectorizer_config=None,  # We'll provide our own vectors
            properties=[
                wvc.config.Property(
                    name="text",
                    data_type=wvc.config.DataType.TEXT
                ),
                wvc.config.Property(
                    name="documentId",
                    data_type=wvc.config.DataType.TEXT
                ),
                wvc.config.Property(
                    name="filename",
                    data_type=wvc.config.DataType.TEXT
                ),
                wvc.config.Property(
                    name="chunkIndex",
                    data_type=wvc.config.DataType.INT
                ),
                wvc.config.Property(
                    name="metadata",
                    data_type=wvc.config.DataType.TEXT
                )
            ]
        )
    else:
        collection = client.collections.get("Document")

except Exception as e:
    print(f"Warning: Error during Weaviate initialization: {str(e)}")
    print("Attempting to connect with fallback configuration...")
    # Fallback to basic configuration
    client = weaviate.connect_to_local(
        skip_init_checks=True,  # Skip all initialization checks
    )
    collection = client.collections.get("Document")

class DocumentProcessor:
    def __init__(self, artifacts_path: Optional[str] = None, enable_remote_services: bool = False):
        """Initialize the DocumentProcessor with advanced configuration options.
        
        Args:
            artifacts_path: Optional path to pre-downloaded model artifacts for offline use
            enable_remote_services: Whether to allow usage of remote services for enhanced processing
        """
        # Configure PDF pipeline options
        pdf_pipeline_options = PdfPipelineOptions(
            artifacts_path=artifacts_path,
            enable_remote_services=enable_remote_services,
            do_table_structure=True,
            do_ocr=True  # Enable OCR for scanned documents
        )
        pdf_pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        
        # Initialize DocumentConverter with format-specific options
        self.doc_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_pipeline_options
                )
            }
        )
        
        # Initialize HybridChunker with a suitable tokenizer
        self.chunker = HybridChunker(tokenizer="BAAI/bge-small-en-v1.5")
        
        # Initialize text splitter for non-docling formats with more lenient settings
        self.text_splitter = CharacterTextSplitter(
            separator="\n",  # More lenient separator
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
            strip_whitespace=True,  # Strip whitespace to avoid empty chunks
            add_start_index=True    # Add start index to help with debugging
        )
        
        self.collection = collection

    def _save_chunks_to_file(self, chunks: List[Any], original_file: str, chunk_meta: Dict = None) -> str:
        """Save chunks to a file for verification."""
        chunks_dir = os.path.join(os.getcwd(), "chunks")
        os.makedirs(chunks_dir, exist_ok=True)
        
        file_base = os.path.splitext(os.path.basename(original_file))[0]
        chunks_file = os.path.join(chunks_dir, f"{file_base}_chunks.txt")
        
        with open(chunks_file, 'w', encoding='utf-8') as f:
            f.write(f"Chunks from {os.path.basename(original_file)}\n")
            f.write("=" * 80 + "\n\n")
            
            for i, chunk in enumerate(chunks):
                f.write(f"Chunk {i}:\n")
                f.write("-" * 40 + "\n")
                
                # Get text and metadata
                if hasattr(chunk, 'text') and hasattr(chunk, 'meta'):
                    text = chunk.text
                    chunk_meta = chunk.meta
                    
                    # Extract metadata
                    metadata_info = []
                    if hasattr(chunk_meta, 'doc_items') and chunk_meta.doc_items:
                        page_numbers = set()
                        element_types = set()
                        
                        for item in chunk_meta.doc_items:
                            if hasattr(item, 'prov'):
                                for prov in item.prov:
                                    if hasattr(prov, 'page_no'):
                                        page_numbers.add(prov.page_no)
                            if hasattr(item, 'label'):
                                element_types.add(str(item.label))
                        
                        if page_numbers:
                            metadata_info.append(f"Pages: {sorted(list(page_numbers))}")
                        
                        if hasattr(chunk_meta, 'headings') and chunk_meta.headings:
                            metadata_info.append(f"Headings: {chunk_meta.headings}")
                else:
                    text = str(chunk)
                    metadata_info = []
                
                # Write chunk information
                f.write(f"Length: {len(text)} characters\n")
                if metadata_info:
                    f.write("\nMetadata:\n")
                    f.write("\n".join(metadata_info) + "\n")
                
                f.write("\nContent:\n")
                f.write(text)
                f.write("\n\n" + "=" * 80 + "\n\n")
            
            f.write(f"\nTotal chunks: {len(chunks)}\n")
        
        print(f"Chunks saved to: {chunks_file}")
        return chunks_file

    def process_document(self, file_path: str, document_id: Optional[str] = None) -> str:
        """Process a document and store its embeddings in Weaviate."""
        doc_id = document_id or str(uuid.uuid4())
        filename = os.path.basename(file_path)
        extension = os.path.splitext(filename)[1].lower()
        
        try:
            chunks = []
            # Process based on file extension
            if extension == '.json':
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                text = json.dumps(data, ensure_ascii=False, indent=2)
                chunks = self.text_splitter.split_text(text)
                print(f"Created {len(chunks)} chunks from {filename}")
            
            elif extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read().strip()
                
                if not text:
                    raise ValueError(f"File {filename} is empty")
                
                chunks = self.text_splitter.split_text(text)
                if not chunks:
                    chunks = CharacterTextSplitter(
                        separator="\n\n",
                        chunk_size=1000,
                        chunk_overlap=200,
                        strip_whitespace=True
                    ).split_text(text)
                
                if not chunks:
                    chunks = [text]
                
                print(f"Created {len(chunks)} chunks from {filename}")
            
            else:
                # For supported formats (PDF, DOCX), use Docling
                result = self.doc_converter.convert(file_path)
                doc = result.document
                chunks = list(self.chunker.chunk(doc))
                print(f"Created {len(chunks)} chunks from {filename}")
            
            # Save chunks to file for verification
            self._save_chunks_to_file(chunks, file_path)
            
            # Delete existing document if it exists
            if document_id:
                self._delete_existing_document(document_id)
            
            # Store chunks with embeddings
            batch_objects = []
            for i, chunk in enumerate(chunks):
                try:
                    # Handle DocChunk objects
                    if hasattr(chunk, 'text') and hasattr(chunk, 'meta'):
                        text = chunk.text
                        chunk_meta = chunk.meta
                        
                        # Initialize metadata dictionary
                        metadata_dict = {
                            "chunk_index": i,
                            "source_type": extension[1:].upper(),
                        }
                        
                        # Extract page numbers from doc_items
                        if hasattr(chunk_meta, 'doc_items') and chunk_meta.doc_items:
                            page_numbers = set()
                            element_types = set()
                            
                            for item in chunk_meta.doc_items:
                                if hasattr(item, 'prov'):
                                    for prov in item.prov:
                                        if hasattr(prov, 'page_no'):
                                            page_numbers.add(prov.page_no)
                            if hasattr(item, 'label'):
                                element_types.add(str(item.label))
                        
                        if page_numbers:
                            metadata_dict["page_numbers"] = sorted(list(page_numbers))
                        
                        if element_types:
                            metadata_dict["element_types"] = sorted(list(element_types))
                        
                        # Get headings if available
                        if hasattr(chunk_meta, 'headings') and chunk_meta.headings:
                            metadata_dict["headings"] = chunk_meta.headings
                        
                        # Generate embedding and create data object
                        embedding = self._get_embedding(text)
                        data_obj = wvc.data.DataObject(
                            properties={
                                "text": text,
                                "documentId": doc_id,
                                "filename": filename,
                                "chunkIndex": i,
                                "metadata": json.dumps(metadata_dict)
                            },
                            vector=embedding
                        )
                        batch_objects.append(data_obj)
                    else:
                        text = str(chunk)
                        embedding = self._get_embedding(text)
                        data_obj = wvc.data.DataObject(
                            properties={
                                "text": text,
                                "documentId": doc_id,
                                "filename": filename,
                                "chunkIndex": i,
                                "metadata": json.dumps({
                                    "chunk_index": i,
                                    "source_type": extension[1:].upper()
                                })
                            },
                            vector=embedding
                        )
                        batch_objects.append(data_obj)
                
                except Exception as e:
                    print(f"Error processing chunk {i}: {str(e)}")
                    continue
            
            if not batch_objects:
                raise ValueError(f"No valid chunks were created for {filename}")
            
            # Use batch insert for better performance
            try:
                result = self.collection.data.insert_many(batch_objects)
                if hasattr(result, 'errors') and result.errors:
                    print(f"Errors during batch insert of {filename}:")
                    for error in result.errors:
                        print(f"  - {error}")
            except Exception as e:
                print(f"Error during batch insert of {filename}: {str(e)}")
                raise
            
            print(f"Successfully processed {filename} with {len(chunks)} chunks")
            return doc_id
            
        except Exception as e:
            print(f"Error processing document {filename}: {str(e)}")
            raise

    def _create_docling_document(self, file_path: str, extension: str) -> DoclingDocument:
        """Create a Docling Document from text or JSON file for consistent chunking.
        
        Args:
            file_path: Path to the file
            extension: File extension (.txt or .json)
            
        Returns:
            DoclingDocument: A Docling Document object
        """
        try:
            filename = os.path.basename(file_path)
            
            if extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                # Create a document with proper structure
                doc = DoclingDocument(name=filename)
                
                # Add main text as a text item
                main_text = TextItem(
                    text=text,
                    label=DocItemLabel.PARAGRAPH,
                    self_ref="#/texts/0",
                    orig=text
                )
                doc.texts.append(main_text)
                
                # Create body structure
                body_node = NodeItem(
                    self_ref="#/body",
                    children=[{"$ref": "#/texts/0"}]
                )
                doc.body = body_node
                
                return doc
            
            elif extension == '.json':
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                
                # Create document with proper structure
                doc = DoclingDocument(name=filename)
                
                # Add title as first text item
                title = TextItem(
                    text="JSON Document",
                    label=DocItemLabel.TITLE,
                    self_ref="#/texts/0",
                    orig="JSON Document"
                )
                doc.texts.append(title)
                
                # Initialize body structure
                body_node = NodeItem(
                    self_ref="#/body",
                    children=[{"$ref": "#/texts/0"}]  # Reference to title
                )
                
                def process_value(key: str, value: Any, parent_node: NodeItem) -> List[dict]:
                    """Process a JSON value and return list of references to created items."""
                    refs = []
                    
                    # Create section heading
                    key_str = str(key)
                    heading = TextItem(
                        text=key_str,
                        label=DocItemLabel.TITLE,
                        self_ref=f"#/texts/{len(doc.texts)}",
                        orig=key_str
                    )
                    doc.texts.append(heading)
                    heading_ref = {"$ref": f"#/texts/{len(doc.texts)-1}"}
                    refs.append(heading_ref)
                    
                    # Create content
                    if isinstance(value, (dict, list)):
                        # For complex values, create a formatted text block
                        formatted_text = json.dumps(value, ensure_ascii=False, indent=2)
                        content = TextItem(
                            text=formatted_text,
                            label=DocItemLabel.TEXT,
                            self_ref=f"#/texts/{len(doc.texts)}",
                            orig=formatted_text
                        )
                    else:
                        # For simple values, create a paragraph
                        value_str = str(value)
                        content = TextItem(
                            text=value_str,
                            label=DocItemLabel.PARAGRAPH,
                            self_ref=f"#/texts/{len(doc.texts)}",
                            orig=value_str
                        )
                    
                    doc.texts.append(content)
                    content_ref = {"$ref": f"#/texts/{len(doc.texts)-1}"}
                    refs.append(content_ref)
                    
                    return refs
                
                if isinstance(data, dict):
                    # Process dictionary items
                    for key, value in data.items():
                        refs = process_value(key, value, body_node)
                        body_node.children.extend(refs)
                
                elif isinstance(data, list):
                    # Process list items
                    for i, item in enumerate(data, 1):
                        refs = process_value(f"Item {i}", item, body_node)
                        body_node.children.extend(refs)
                
                else:
                    # Handle simple value
                    value_str = str(data)
                    content = TextItem(
                        text=value_str,
                        label=DocItemLabel.PARAGRAPH,
                        self_ref=f"#/texts/{len(doc.texts)}",
                        orig=value_str
                    )
                    doc.texts.append(content)
                    body_node.children.append({"$ref": f"#/texts/{len(doc.texts)-1}"})
                
                # Set body
                doc.body = body_node
                
                print(f"Created DoclingDocument with {len(doc.texts)} text items")
                return doc
            
        except Exception as e:
            print(f"Error details for {file_path}:")
            print(f"Exception type: {type(e)}")
            print(f"Exception args: {e.args}")
            raise ValueError(f"Error creating Docling document from {extension} file: {str(e)}")

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding from OpenAI API."""
        response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    def _delete_existing_document(self, doc_id: str):
        """Delete existing document from Weaviate."""
        self.collection.data.delete_many(
            where=wvc.query.Filter.by_property("documentId").equal(doc_id)
        ) 