# Design Document: Aawaz 2.0 – Voice-First Civic AI for Rural India

## Overview

I am Aawaz 2.0, a voice-first civic assistance platform that combines Automatic Speech Recognition (ASR), Retrieval-Augmented Generation (RAG), and Text-to-Speech (TTS) to provide accessible government scheme information to rural citizens in Assam, India. My architecture follows a modular pipeline design where user voice input flows through language processing, semantic search, LLM reasoning, and voice output generation.

My core innovation lies in the RAG pipeline that grounds LLM responses in verified government documents, preventing hallucination while enabling natural language interaction in local languages. I support two primary interaction modes: voice-to-voice conversation for queries and image-based document simplification (Snap & Explain) for physical documents.

My key design principles:
- **Voice-First**: Optimized for spoken interaction with minimal UI complexity
- **Language-Inclusive**: Native support for Assamese, Bodo, and Hindi
- **Accuracy-Grounded**: All responses backed by retrieved government documents
- **Low-Latency**: End-to-end response within 5 seconds
- **Cost-Efficient**: Optimized for hackathon budget constraints

## Architecture

### System Architecture Overview

I follow a three-layer architecture:

**1. Frontend Layer (User Interface)**
- WhatsApp Business API integration for voice messages
- Twilio Voice API for phone-based interaction
- Web interface (React-based) for browser access
- Handles audio capture, upload, and playback

**2. AI Processing Layer (My Core Intelligence)**
- ASR Service: Converts voice to text (Bhashini API or Whisper)
- Language Detection: Identifies input language
- Embedding Service: Generates query embeddings
- Vector Search: Retrieves relevant document chunks
- LLM Engine: Generates contextual responses (Llama 3 or Gemini 1.5 Flash)
- Translation Service: Ensures response in user's language
- TTS Service: Converts text response to speech
- OCR Service: Extracts text from document images

**3. Data Layer (My Knowledge Management)**
- Vector Database: Stores document embeddings (Pinecone or Milvus)
- Document Store: Raw government scheme PDFs and metadata
- Cache Layer: Frequently accessed responses and embeddings

### Architecture Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│  WhatsApp Business API  │  Twilio Voice  │  Web Interface       │
│  (Voice Messages)       │  (Phone Calls) │  (Browser Access)    │
└────────────┬────────────┴────────┬───────┴──────────┬───────────┘
             │                     │                   │
             └─────────────────────┼───────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AI PROCESSING LAYER                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────┐      ┌──────────────┐      ┌─────────────┐       │
│  │   ASR    │─────▶│   Language   │─────▶│  Embedding  │       │
│  │ Service  │      │  Detection   │      │   Service   │       │
│  └──────────┘      └──────────────┘      └──────┬──────┘       │
│                                                   │              │
│                                                   ▼              │
│                                          ┌─────────────┐        │
│                                          │   Vector    │        │
│                                          │   Search    │        │
│                                          └──────┬──────┘        │
│                                                 │                │
│                                                 ▼                │
│  ┌──────────┐      ┌──────────────┐      ┌─────────────┐       │
│  │   TTS    │◀─────│ Translation  │◀─────│     LLM     │       │
│  │ Service  │      │   Service    │      │   Engine    │       │
│  └──────────┘      └──────────────┘      └─────────────┘       │
│                                                                   │
│  ┌──────────────────────────────────────────────────────┐       │
│  │         SNAP & EXPLAIN MODULE                        │       │
│  │  Image Upload ─▶ OCR ─▶ Simplification ─▶ TTS       │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                   │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│  Vector Database  │  Document Store  │  Cache Layer             │
│  (Pinecone/Milvus)│  (S3/GCS)       │  (Redis)                 │
└─────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. ASR Service Component

**Purpose**: Convert user voice input to text

**Interface**:
```
Input: 
  - audio_data: bytes (WAV, MP3, or OGG format)
  - sample_rate: integer (default 16000 Hz)
  
Output:
  - transcribed_text: string
  - detected_language: string (ISO 639-1 code)
  - confidence_score: float (0.0 to 1.0)
```

**Implementation Options**:
- **Bhashini API**: Government-backed ASR with native Assamese/Bodo support
- **OpenAI Whisper**: Multilingual model with good accuracy
- **Google Speech-to-Text**: Commercial option with language detection

**Error Handling**:
- Low confidence (<0.6): Request user to repeat
- Unsupported language: Fallback to Hindi or English
- Audio quality issues: Return error with guidance

### 2. Embedding Service Component

**Purpose**: Generate vector embeddings for queries and documents

**Interface**:
```
Input:
  - text: string
  - language: string
  
Output:
  - embedding: float[] (384 or 768 dimensions)
```

**Implementation**:
- **Model**: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
- **Dimension**: 768
- **Normalization**: L2 normalized for cosine similarity

### 3. Vector Search Component

**Purpose**: Retrieve relevant document chunks based on semantic similarity

**Interface**:
```
Input:
  - query_embedding: float[]
  - top_k: integer (default 5)
  - similarity_threshold: float (default 0.7)
  - filters: dict (optional metadata filters)
  
Output:
  - results: list of {
      chunk_text: string,
      similarity_score: float,
      metadata: {
        document_name: string,
        scheme_name: string,
        source_url: string,
        last_updated: date
      }
    }
```

**Vector Database Options**:
- **Pinecone**: Managed service, easy setup, free tier available
- **Milvus**: Open-source, self-hosted, more control
- **Qdrant**: Lightweight alternative with good performance

### 4. LLM Engine Component

**Purpose**: Generate contextual responses using retrieved documents

**Interface**:
```
Input:
  - user_query: string
  - retrieved_context: list of strings
  - language: string
  - max_tokens: integer (default 500)
  
Output:
  - response: string
  - sources_used: list of document names
```

**Prompt Template**:
```
I am Aawaz, a helpful civic assistant for rural India. I answer user questions based ONLY on the provided context. If the context doesn't contain the answer, I say so clearly.

Context:
{retrieved_context}

User Question: {user_query}

Instructions:
- Answer in {language}
- Use simple language suitable for low literacy users
- Include eligibility criteria if mentioned
- Provide step-by-step application instructions if available
- Cite the source document name

Answer:
```

**Model Options**:
- **Llama 3 8B**: Open-source, good multilingual support, can run on modest hardware
- **Gemini 1.5 Flash**: Fast, cost-effective, good for hackathon
- **GPT-3.5 Turbo**: Reliable fallback option

### 5. TTS Service Component

**Purpose**: Convert text responses to natural speech

**Interface**:
```
Input:
  - text: string
  - language: string
  - voice_gender: string (male/female)
  
Output:
  - audio_data: bytes (MP3 format)
  - duration_seconds: float
```

**Implementation Options**:
- **Google Cloud TTS**: Good multilingual support including Indic languages
- **Azure TTS**: Neural voices with natural prosody
- **Bhashini TTS**: Government service for Indic languages

### 6. OCR Service Component (Snap & Explain)

**Purpose**: Extract text from document images

**Interface**:
```
Input:
  - image_data: bytes (JPEG, PNG)
  - language_hint: string (optional)
  
Output:
  - extracted_text: string
  - detected_language: string
  - confidence_score: float
  - bounding_boxes: list (optional, for layout analysis)
```

**Implementation**:
- **Tesseract OCR**: Open-source, supports Indic scripts
- **Google Cloud Vision API**: High accuracy, language detection
- **PaddleOCR**: Good for multilingual documents

### 7. Document Simplification Component

**Purpose**: Convert complex government documents to simple explanations

**Interface**:
```
Input:
  - document_text: string
  - target_language: string
  - reading_level: string (default "basic")
  
Output:
  - simplified_summary: string
  - key_points: list of strings
  - action_steps: list of strings
```

**Implementation**:
- Use LLM with specific simplification prompt
- Extract structured information (eligibility, documents needed, application process)
- Translate to target language if needed

## Data Models

### Document Metadata Schema

```json
{
  "document_id": "string (UUID)",
  "document_name": "string",
  "scheme_name": "string",
  "category": "string (e.g., 'financial_assistance', 'healthcare', 'housing')",
  "source_url": "string",
  "source_organization": "string (e.g., 'Assam Government', 'MyGov.in')",
  "language": "string",
  "last_updated": "ISO 8601 date",
  "file_path": "string (S3/GCS path)",
  "total_chunks": "integer",
  "indexed_date": "ISO 8601 date"
}
```

### Chunk Schema

```json
{
  "chunk_id": "string (UUID)",
  "document_id": "string (foreign key)",
  "chunk_index": "integer",
  "text": "string (500-1000 tokens)",
  "embedding": "float[] (768 dimensions)",
  "metadata": {
    "document_name": "string",
    "scheme_name": "string",
    "page_number": "integer",
    "section": "string"
  }
}
```

### Query Log Schema

```json
{
  "query_id": "string (UUID)",
  "timestamp": "ISO 8601 datetime",
  "user_id": "string (anonymized)",
  "input_language": "string",
  "query_text": "string",
  "retrieved_chunks": "list of chunk_ids",
  "response_text": "string",
  "response_time_ms": "integer",
  "user_feedback": "string (optional)"
}
```

### User Session Schema

```json
{
  "session_id": "string (UUID)",
  "channel": "string (whatsapp/twilio/web)",
  "start_time": "ISO 8601 datetime",
  "end_time": "ISO 8601 datetime",
  "preferred_language": "string",
  "query_count": "integer"
}
```

## RAG Pipeline Design

### Document Ingestion Process

**Step 1: Document Collection**
- Scrape PDFs from MyGov.in, Assam.gov.in
- Download scheme documents (Orunodoi, PMAY, Ration Card, etc.)
- Store raw PDFs in document store (S3/GCS)

**Step 2: Text Extraction**
- Use PyPDF2 or pdfplumber to extract text from PDFs
- Preserve document structure and metadata
- Handle multi-column layouts and tables

**Step 3: Chunking Strategy**
- **Chunk Size**: 500-1000 tokens (approximately 2-4 paragraphs)
- **Overlap**: 100 tokens between consecutive chunks to preserve context
- **Boundary Detection**: Split on paragraph or section boundaries, not mid-sentence
- **Metadata Preservation**: Each chunk retains document name, scheme name, page number

**Step 4: Embedding Generation**
- Generate embeddings using multilingual sentence transformer
- Normalize embeddings for cosine similarity
- Batch process for efficiency (32-64 chunks per batch)

**Step 5: Vector Database Indexing**
- Store embeddings with metadata in vector database
- Create indexes for fast similarity search
- Set up metadata filters (category, language, date)

### Query Processing Flow

**Step 1: Voice to Text**
- Receive audio from user
- ASR converts to text
- Detect language automatically

**Step 2: Query Embedding**
- Generate embedding for user query
- Use same model as document embeddings
- Normalize for similarity comparison

**Step 3: Similarity Search**
- Query vector database with embedding
- Retrieve top 5 chunks with similarity > 0.7
- Apply metadata filters if needed (e.g., specific scheme category)

**Step 4: Context Preparation**
- Concatenate retrieved chunks
- Add document source information
- Limit total context to 3000 tokens to fit LLM window

**Step 5: LLM Response Generation**
- Inject context and query into prompt template
- Generate response with source attribution
- Ensure response stays grounded in context

**Step 6: Translation (if needed)**
- If LLM generates in different language than query, translate
- Preserve technical terms and scheme names

**Step 7: Text to Speech**
- Convert response to audio in query language
- Return audio to user

### Caching Strategy

**Query Cache**:
- Cache common queries and responses
- Key: hash of (query_text + language)
- TTL: 24 hours
- Reduces LLM API calls by ~40%

**Embedding Cache**:
- Cache embeddings for frequent queries
- Reduces embedding API calls

**Response Cache**:
- Cache complete voice responses for identical queries
- Reduces TTS API calls

## Snap & Explain Design

### Image Processing Pipeline

**Step 1: Image Upload**
- Accept JPEG/PNG images up to 10MB
- Validate image format and size
- Compress if needed to reduce processing time

**Step 2: OCR Processing**
- Extract text using OCR service
- Detect language of extracted text
- Return confidence score

**Step 3: Language Detection and Translation**
- Identify document language
- If different from user's preferred language, translate

**Step 4: Document Analysis**
- Use LLM to analyze document type (application form, scheme notice, etc.)
- Extract key information: scheme name, eligibility, required documents, deadlines

**Step 5: Simplification**
- Generate plain-language summary
- Create numbered action steps
- Highlight important dates and requirements

**Step 6: Output Generation**
- Present simplified text in user's language
- Optionally convert to speech
- Provide option to ask follow-up questions

### Vision-Capable LLM Integration

For complex documents with tables and forms, use vision-capable LLM:
- **GPT-4 Vision**: Excellent understanding of document layout
- **Gemini 1.5 Pro**: Good balance of cost and capability
- **Claude 3**: Strong document analysis capabilities

**Prompt Template for Document Analysis**:
```
I am analyzing a government document image for a rural citizen with low literacy. 

Task:
1. Identify what type of document this is
2. Extract the main purpose and key information
3. List eligibility criteria if present
4. List required documents if present
5. Provide step-by-step instructions in simple language

Output in {language} using simple vocabulary.
```

## Technology Stack

### Core Services
- **Backend Framework**: FastAPI (Python) or Express.js (Node.js)
- **ASR**: Bhashini API (primary), OpenAI Whisper (fallback)
- **Embedding Model**: sentence-transformers/paraphrase-multilingual-mpnet-base-v2
- **Vector Database**: Pinecone (managed) or Milvus (self-hosted)
- **LLM**: Llama 3 8B (via Groq API) or Gemini 1.5 Flash
- **TTS**: Google Cloud TTS or Bhashini TTS
- **OCR**: Tesseract OCR or Google Cloud Vision API

### Frontend
- **Web Interface**: React with Material-UI
- **Voice Recording**: Web Audio API
- **WhatsApp Integration**: WhatsApp Business API
- **Phone Integration**: Twilio Voice API

### Infrastructure
- **Cloud Platform**: AWS, GCP, or Azure
- **Compute**: Serverless functions (Lambda/Cloud Functions) or containers (ECS/Cloud Run)
- **Storage**: S3/GCS for documents
- **Cache**: Redis for response caching
- **Monitoring**: CloudWatch/Cloud Logging

### Development Tools
- **API Testing**: Postman
- **Version Control**: Git/GitHub
- **Deployment**: Docker containers
- **CI/CD**: GitHub Actions

## Scalability Strategy

### Horizontal Scaling
- **Stateless Backend**: All API endpoints are stateless
- **Load Balancing**: Distribute requests across multiple instances
- **Auto-scaling**: Scale based on request volume and latency

### Database Scaling
- **Vector Database**: Pinecone handles scaling automatically; Milvus can be clustered
- **Document Store**: S3/GCS scales automatically
- **Cache**: Redis cluster for distributed caching

### Performance Optimization
- **Batch Processing**: Process multiple embeddings in parallel
- **Connection Pooling**: Reuse database connections
- **Async Processing**: Use async/await for I/O operations
- **CDN**: Cache static assets and audio files

### Cost Optimization
- **Model Selection**: Use lightweight models (Llama 3 8B vs larger models)
- **Context Window**: Limit to 4000 tokens to reduce LLM costs
- **Caching**: Aggressive caching of common queries
- **Batch API Calls**: Combine multiple operations where possible

## Security and Privacy

### Data Protection
- **No Permanent Voice Storage**: Audio deleted after transcription
- **Encrypted Transit**: TLS 1.3 for all API communications
- **Encrypted at Rest**: S3/GCS encryption for stored documents
- **Anonymized Logging**: Remove PII from logs

### Access Control
- **API Authentication**: API keys for service-to-service communication
- **Rate Limiting**: Prevent abuse (100 requests per user per hour)
- **Input Validation**: Sanitize all user inputs

### Privacy Compliance
- **Data Retention**: Delete temporary data within 1 hour
- **User Consent**: Clear privacy policy for data usage
- **Minimal Collection**: Only collect data necessary for functionality

## Error Handling

### ASR Errors
- **Low Confidence**: "I couldn't understand clearly. Please speak again."
- **Unsupported Language**: "I currently support Assamese, Bodo, and Hindi. Please try one of these languages."
- **Audio Quality**: "The audio quality is poor. Please try recording in a quieter environment."

### RAG Pipeline Errors
- **No Results Found**: "I don't have information about this in my knowledge base. Please contact [helpline] for assistance."
- **Low Similarity**: "I found some related information, but I'm not confident it answers your question. Here's what I found..."
- **LLM Failure**: "I'm having trouble generating a response right now. Please try again in a moment."

### TTS Errors
- **TTS Unavailable**: Display text response instead of audio
- **Language Not Supported**: Fallback to English or Hindi

### System Errors
- **Database Unavailable**: "The service is temporarily unavailable. Please try again in a few minutes."
- **Timeout**: "This is taking longer than expected. Please try again."

## Testing Strategy

The testing strategy for Aawaz 2.0 combines unit testing for specific examples and edge cases with property-based testing for universal correctness properties. This dual approach ensures both concrete functionality and general correctness across diverse inputs.

### Dual Testing Approach

**Unit Testing** and **Property-Based Testing** are complementary and both are required:

**Unit Tests** focus on:
- **Specific Examples**: Verify correct behavior for known inputs (e.g., "Orunodoi scheme" query in Assamese returns expected information)
- **Edge Cases**: Test boundary conditions (empty audio, very long queries, special characters in document text, poor quality images)
- **Error Conditions**: Validate error handling (network failures, invalid audio formats, service timeouts, malformed responses)
- **Integration Points**: Test component interactions (ASR → Embedding → Vector Search → LLM → TTS pipeline)

**Property-Based Tests** focus on:
- **Universal Properties**: Verify properties that hold across all valid inputs (e.g., response language always matches query language)
- **Comprehensive Coverage**: Test with randomized inputs to discover edge cases not anticipated in unit tests
- **Invariants**: Verify system invariants (e.g., chunk sizes always within bounds, embeddings always normalized)
- **Round-Trip Properties**: Verify operations and their inverses (e.g., embed → search → retrieve returns original chunk)

### Property-Based Testing Configuration

**Library Selection**:
- **Python**: Use Hypothesis for property-based testing
- **JavaScript/TypeScript**: Use fast-check for property-based testing

**Test Configuration**:
- **Minimum 100 iterations** per property test (due to randomization)
- Each property test MUST reference its design document property
- Tag format: `# Feature: Aawaz-civic-ai, Property {number}: {property_text}`

**Example Property Test Structure** (Python with Hypothesis):
```python
from hypothesis import given, strategies as st
import pytest

# Feature: Aawaz-civic-ai, Property 1: Language Consistency
@given(
    query_text=st.text(min_size=10, max_size=200),
    language=st.sampled_from(['as', 'brx', 'hi', 'en'])  # Assamese, Bodo, Hindi, English
)
def test_language_consistency(query_text, language):
    """For any query in a supported language, response SHALL be in same language"""
    response = voice_assistant.process_query(query_text, language)
    detected_response_lang = detect_language(response.text)
    assert detected_response_lang == language
```

**Example Property Test Structure** (JavaScript with fast-check):
```javascript
import fc from 'fast-check';

// Feature: Aawaz-civic-ai, Property 17: Document Chunking Size
test('Property 17: All chunks are between 500-1000 tokens', () => {
  fc.assert(
    fc.property(
      fc.lorem(1000, 5000), // Generate random document text
      (documentText) => {
        const chunks = chunkDocument(documentText);
        return chunks.every(chunk => {
          const tokenCount = countTokens(chunk);
          return tokenCount >= 500 && tokenCount <= 1000;
        });
      }
    ),
    { numRuns: 100 }
  );
});
```

### Testing Tools

- **Unit Testing**: pytest (Python) or Jest (JavaScript/TypeScript)
- **Property Testing**: Hypothesis (Python) or fast-check (JavaScript/TypeScript)
- **API Testing**: Postman/Newman for endpoint validation
- **Load Testing**: Locust or k6 for performance and scalability testing
- **Audio Testing**: librosa (Python) for audio validation
- **Image Testing**: Pillow (Python) or Sharp (JavaScript) for image validation

### Test Coverage Goals

- **Unit Test Coverage**: >80% code coverage for all components
- **Property Test Coverage**: All 46 correctness properties implemented as property tests
- **Integration Test Coverage**: All API endpoints tested with valid and invalid inputs
- **End-to-End Testing**: Complete voice-to-voice and Snap & Explain flows validated
- **Performance Testing**: All latency properties validated under load

### Test Organization

**Unit Tests**:
```
tests/
  unit/
    test_asr_service.py          # ASR component tests
    test_embedding_service.py    # Embedding generation tests
    test_vector_search.py        # Vector database search tests
    test_llm_engine.py          # LLM response generation tests
    test_tts_service.py         # TTS conversion tests
    test_ocr_service.py         # OCR extraction tests
    test_document_chunking.py   # Document processing tests
```

**Property Tests**:
```
tests/
  properties/
    test_language_properties.py      # Properties 1, 3, 10, 45
    test_rag_properties.py          # Properties 4, 5, 6, 46
    test_response_properties.py     # Properties 7, 8, 9
    test_document_properties.py     # Properties 12-19
    test_performance_properties.py  # Properties 20-24
    test_security_properties.py     # Properties 31-34
    test_error_properties.py        # Properties 35-39
```

**Integration Tests**:
```
tests/
  integration/
    test_voice_pipeline.py          # End-to-end voice flow
    test_snap_explain_pipeline.py   # End-to-end image flow
    test_api_endpoints.py          # All REST API endpoints
```

### Edge Cases to Test

Based on prework analysis, these edge cases require special attention:

1. **Poor Audio Quality** (1.3): Test with noisy audio, low volume, background noise
2. **Mixed Language Input** (1.4): Test with code-switching between languages
3. **Poor Image Quality** (6.6): Test with blurry, low-resolution, skewed document images
4. **Empty Inputs**: Test with empty strings, silent audio, blank images
5. **Very Long Inputs**: Test with audio >5 minutes, documents >100 pages
6. **Special Characters**: Test with Unicode, emojis, non-standard characters
7. **Network Failures**: Test with timeouts, connection drops, slow responses

### Continuous Testing

- **Pre-commit Hooks**: Run unit tests before each commit
- **CI/CD Pipeline**: Run all tests on pull requests
- **Nightly Builds**: Run property tests with higher iteration counts (1000+)
- **Performance Monitoring**: Track latency metrics in production

## Hackathon Demo Architecture

### Functional Components (Fully Implemented)
- Voice input capture (web interface)
- ASR using Whisper or Bhashini
- Embedding generation
- Vector search with Pinecone
- LLM response generation with Gemini 1.5 Flash
- TTS output
- Basic web UI

### Simplified Components (MVP Version)
- **Knowledge Base**: 20-30 key schemes instead of comprehensive coverage
- **Languages**: Focus on Assamese and Hindi, Bodo as stretch goal
- **Channels**: Web interface primary, WhatsApp as demo only

### Mocked/Simulated Components
- **User Authentication**: Simple session IDs, no real auth
- **Analytics**: Basic logging, no dashboard
- **Feedback System**: Placeholder only

### Demo Flow
1. User opens web interface
2. Clicks microphone button and speaks query in Assamese
3. System transcribes, searches knowledge base, generates response
4. Response played back in Assamese
5. User uploads document image
6. System extracts text, simplifies, and explains in Assamese

### Success Metrics for Demo
- End-to-end response time < 5 seconds
- Correct language detection and response
- Accurate scheme information retrieval
- Clear, simple language in responses
- Successful Snap & Explain demonstration


## Correctness Properties

A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

### Property Reflection

After analyzing all acceptance criteria, I identified several opportunities to consolidate redundant properties:

**Language Consistency (4.1-4.4)**: These four criteria all test the same property—that response language matches query language. Combined into Property 1.

**Performance Timing (8.1-8.5)**: While these test different components, they all verify latency bounds. Kept separate as they test distinct components with different thresholds.

**Channel Availability (9.1-9.3)**: These are specific examples of channel support rather than universal properties. Kept as examples.

**Error Handling (12.1-12.5)**: Each tests different failure modes. Kept separate as they validate distinct error scenarios.

**Chunk Size and Embedding (7.2-7.3)**: These are sequential steps in the same pipeline but test different properties. Kept separate.

### Core Functional Properties

**Property 1: Language Consistency**
*For any* query in a supported language (Assamese, Bodo, Hindi, or English), I SHALL generate my response in the same language as the query.
**Validates: Requirements 4.1, 4.2, 4.3, 4.4**

**Property 2: Audio Format Compatibility**
*For any* valid audio input in WAV, MP3, or OGG format, I SHALL successfully accept and process the audio.
**Validates: Requirements 1.5**

**Property 3: Automatic Language Detection**
*For any* audio input in a supported language, I SHALL correctly detect the language without explicit language specification.
**Validates: Requirements 1.2**

**Property 4: RAG Pipeline Execution**
*For any* user query, I SHALL generate an embedding and execute a vector database search.
**Validates: Requirements 2.1**

**Property 5: Top-K Retrieval**
*For any* query with matching results, I SHALL retrieve at most 5 document chunks, ordered by descending similarity score.
**Validates: Requirements 2.2, 2.4**

**Property 6: No-Results Handling**
*For any* query where all similarity scores are below 0.7, I SHALL return a message indicating that information is not available.
**Validates: Requirements 2.3**

**Property 7: Context Grounding**
*For any* generated response, all factual claims SHALL be present in the retrieved context chunks, preventing hallucination.
**Validates: Requirements 3.2, 14.3**

**Property 8: Eligibility Criteria Inclusion**
*For any* context containing eligibility criteria, I SHALL include those criteria in a clearly formatted list in my generated response.
**Validates: Requirements 3.1, 3.3**

**Property 9: Action Steps Formatting**
*For any* context containing application steps, I SHALL present them as numbered action steps in my generated response.
**Validates: Requirements 3.4**

**Property 10: TTS Language Matching**
*For any* text response in a supported language, I SHALL generate audio in that same language.
**Validates: Requirements 5.1**

**Property 11: Audio Output Format**
*For any* generated audio output, the format SHALL be compatible with common mobile devices (MP3 or similar standard format).
**Validates: Requirements 5.5**

**Property 12: OCR Text Extraction**
*For any* valid document image, I SHALL extract text content from the image.
**Validates: Requirements 6.1**

**Property 13: Document Language Detection**
*For any* extracted text from a document, I SHALL detect the language of the text.
**Validates: Requirements 6.2**

**Property 14: Document Translation**
*For any* document in a detected language, I SHALL translate it to the user's preferred language.
**Validates: Requirements 6.3**

**Property 15: Document Summarization**
*For any* processed document, I SHALL generate a simplified summary.
**Validates: Requirements 6.4**

**Property 16: Action Steps Extraction**
*For any* document containing application instructions, I SHALL extract and format them as action steps.
**Validates: Requirements 6.5**

**Property 17: Document Chunking Size**
*For any* document added to my knowledge base, all generated chunks SHALL be between 500 and 1000 tokens in size.
**Validates: Requirements 7.2**

**Property 18: Embedding Generation**
*For any* document chunk, I SHALL generate a vector embedding.
**Validates: Requirements 7.3**

**Property 19: Metadata Storage**
*For any* stored embedding, my vector database entry SHALL include metadata fields: document_name, scheme_name, source_url, and last_updated.
**Validates: Requirements 7.4**

### Performance Properties

**Property 20: ASR Latency**
*For any* audio input, I SHALL complete speech-to-text conversion within 3 seconds.
**Validates: Requirements 1.1, 8.2**

**Property 21: Vector Search Latency**
*For any* query embedding, my vector database search SHALL complete within 1 second.
**Validates: Requirements 8.3**

**Property 22: LLM Response Latency**
*For any* query with retrieved context, I SHALL generate a response within 2 seconds.
**Validates: Requirements 8.4**

**Property 23: End-to-End Voice Latency**
*For any* voice query, I SHALL deliver the complete voice-to-voice response within 5 seconds.
**Validates: Requirements 5.4, 8.1**

**Property 24: Image Processing Latency**
*For any* uploaded document image, I SHALL provide a summary within 8 seconds.
**Validates: Requirements 8.5**

### Interface and Accessibility Properties

**Property 25: Multi-Input Support**
*For any* web interface session, I SHALL accept both voice input and text input.
**Validates: Requirements 9.4**

### Optimization Properties

**Property 26: Audio Compression Under Low Bandwidth**
*For any* network condition with bandwidth below 100 kbps, I SHALL compress audio to maintain functionality.
**Validates: Requirements 10.1**

**Property 27: Efficient Audio Codec Usage**
*For any* voice data transmission, I SHALL use efficient audio codecs to minimize data transfer.
**Validates: Requirements 10.2**

**Property 28: Latency Indicator Display**
*For any* request where network latency exceeds 2 seconds, I SHALL display a waiting indicator.
**Validates: Requirements 10.3**

**Property 29: Query Caching**
*For any* query that has been processed before, I SHALL serve the response from cache on subsequent identical queries.
**Validates: Requirements 10.4**

**Property 30: Request Queuing**
*For any* request made during intermittent connectivity, I SHALL queue the request and process it when connection is restored.
**Validates: Requirements 10.5**

### Security and Privacy Properties

**Property 31: Audio Non-Persistence**
*For any* processed voice input, I SHALL delete the audio recording after transcription is complete.
**Validates: Requirements 11.1**

**Property 32: Encrypted API Communication**
*For any* API call, the connection SHALL use TLS encryption (HTTPS).
**Validates: Requirements 11.2**

**Property 33: Log Anonymization**
*For any* logged interaction, I SHALL anonymize or hash user identifiers.
**Validates: Requirements 11.3**

**Property 34: Temporary Data Cleanup**
*For any* session that has ended, I SHALL delete all temporary audio and text data within 1 hour.
**Validates: Requirements 11.5**

### Error Handling Properties

**Property 35: ASR Failure Handling**
*For any* ASR processing failure, I SHALL return an error message in the user's local language requesting retry.
**Validates: Requirements 12.1**

**Property 36: Database Unavailability Handling**
*For any* vector database unavailability, I SHALL return a message indicating temporary service unavailability.
**Validates: Requirements 12.2**

**Property 37: LLM Failure Fallback**
*For any* LLM generation failure, I SHALL provide a fallback message with contact information.
**Validates: Requirements 12.3**

**Property 38: TTS Failure Fallback**
*For any* TTS service failure, I SHALL display the response as text instead of audio.
**Validates: Requirements 12.4**

**Property 39: Error Logging Completeness**
*For any* component failure, I SHALL log the error with timestamp, component name, error type, and error message.
**Validates: Requirements 12.5**

### Extensibility Properties

**Property 40: Language Configuration Extensibility**
*For any* new language added through configuration, I SHALL support queries and responses in that language.
**Validates: Requirements 13.5**

### Content Accuracy Properties

**Property 41: Source Attribution**
*For any* response providing scheme information, I SHALL include the source document name.
**Validates: Requirements 14.1**

**Property 42: Eligibility Citation**
*For any* response stating eligibility criteria, I SHALL cite the official scheme guidelines.
**Validates: Requirements 14.2**

**Property 43: Uncertainty Disclosure**
*For any* response with low confidence (similarity score < 0.8), I SHALL include a disclaimer to verify with official sources.
**Validates: Requirements 14.4**

**Property 44: Document Metadata Completeness**
*For any* document in my knowledge base, the metadata SHALL include a last_updated date field.
**Validates: Requirements 14.5**

### Round-Trip Properties

**Property 45: Query-Response Language Round-Trip**
*For any* query in language L, generating a response and detecting the response language SHALL return language L.
**Validates: Requirements 4.1, 4.2, 4.3, 4.4**

**Property 46: Embedding-Search Round-Trip**
*For any* document chunk, generating its embedding, storing it, and searching with the same embedding SHALL return that chunk as the top result.
**Validates: Requirements 2.1, 7.3, 7.4**