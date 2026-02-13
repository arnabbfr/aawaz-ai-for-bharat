# Requirements Document: Aawaz – Voice-First Civic AI for Rural India

## Introduction

Aawaz is a voice-first civic assistance system designed to bridge the digital divide for rural citizens in Assam, India. The system addresses the critical challenge of accessing government schemes and civic information for users with low digital literacy, elderly citizens, and those who prefer local language interaction. By leveraging voice interaction, document simplification, and multilingual support, Sahayak 2.0 enables equitable access to essential government services and information.

The system targets rural populations in Assam with scalability potential across India, focusing on users who face barriers due to language, literacy, or technological complexity. Core capabilities include voice-to-voice civic assistance, document simplification through image capture, and a comprehensive knowledge base of government schemes.

## Glossary

- **Sahayak_System**: The complete voice-first civic AI platform including voice interface, RAG pipeline, and knowledge base
- **Voice_Assistant**: The voice-to-voice interaction component handling ASR, LLM reasoning, and TTS
- **Snap_Explain_Module**: The image-based document processing component using OCR and simplification
- **Knowledge_Base**: The indexed repository of government schemes, policies, and civic information
- **RAG_Pipeline**: Retrieval-Augmented Generation pipeline combining vector search with LLM reasoning
- **ASR_Service**: Automatic Speech Recognition service converting voice to text
- **TTS_Service**: Text-to-Speech service converting text responses to audio
- **Vector_Database**: Database storing document embeddings for similarity search
- **LLM_Engine**: Large Language Model processing queries and generating responses
- **OCR_Service**: Optical Character Recognition service extracting text from images
- **User**: Rural citizen, elderly person, or individual with low digital literacy seeking civic information
- **Government_Scheme**: Official program, policy, or service provided by government authorities
- **Local_Language**: Assamese, Bodo, Hindi, or other regional languages spoken in target areas
- **Query**: User request for information about government schemes or civic services
- **Response**: System-generated answer providing scheme information and actionable guidance
- **Document**: Government scheme PDF, policy document, or official civic information source
- **Chunk**: Segmented portion of a document used for embedding and retrieval
- **Embedding**: Vector representation of text used for semantic similarity search
- **Context**: Retrieved document chunks relevant to user query
- **Action_Step**: Specific guidance on how to apply for or access a government scheme

## Requirements

### Requirement 1: Voice Input and Processing

**User Story:** As a rural citizen with low digital literacy, I want to ask questions using my voice in my local language, so that I can access civic information without typing or reading complex text.

#### Acceptance Criteria

1. WHEN a User speaks a Query in Assamese, Bodo, or Hindi, THE ASR_Service SHALL convert the speech to text within 3 seconds
2. WHEN the ASR_Service receives audio input, THE Sahayak_System SHALL detect the language automatically
3. WHEN audio quality is poor or speech is unclear, THE ASR_Service SHALL request the User to repeat the Query
4. WHEN a User speaks in mixed languages, THE ASR_Service SHALL process the dominant language and normalize the text
5. THE Voice_Assistant SHALL accept audio input in common formats including WAV, MP3, and OGG

### Requirement 2: Knowledge Retrieval and RAG Pipeline

**User Story:** As a User, I want the system to find relevant government scheme information from official sources, so that I receive accurate and trustworthy answers to my questions.

#### Acceptance Criteria

1. WHEN a Query is received, THE RAG_Pipeline SHALL generate an embedding and search the Vector_Database for relevant Document chunks
2. WHEN relevant Context is found, THE RAG_Pipeline SHALL retrieve the top 5 most similar chunks based on semantic similarity
3. WHEN no relevant Context is found with similarity score above 0.7, THE Sahayak_System SHALL inform the User that information is not available
4. WHEN multiple relevant schemes match a Query, THE RAG_Pipeline SHALL rank results by relevance score
5. THE Vector_Database SHALL index Documents from MyGov.in, Assam.gov.in, and official scheme PDFs

### Requirement 3: Response Generation and Reasoning

**User Story:** As a User, I want to receive clear, actionable answers in my local language, so that I understand what schemes I'm eligible for and how to apply.

#### Acceptance Criteria

1. WHEN Context is retrieved, THE LLM_Engine SHALL generate a Response that includes scheme eligibility criteria and application steps
2. WHEN generating a Response, THE LLM_Engine SHALL use only the retrieved Context and avoid hallucination
3. WHEN a scheme has specific eligibility requirements, THE Response SHALL list them clearly
4. WHEN application steps exist, THE Response SHALL provide them as numbered Action_Steps
5. THE LLM_Engine SHALL generate Responses in simple language appropriate for low literacy users

### Requirement 4: Multilingual Support and Translation

**User Story:** As a User who speaks Assamese, Bodo, or Hindi, I want to receive answers in my preferred language, so that I can fully understand the information provided.

#### Acceptance Criteria

1. WHEN a Query is processed in Assamese, THE Sahayak_System SHALL generate the Response in Assamese
2. WHEN a Query is processed in Bodo, THE Sahayak_System SHALL generate the Response in Bodo
3. WHEN a Query is processed in Hindi, THE Sahayak_System SHALL generate the Response in Hindi
4. WHERE English is requested, THE Sahayak_System SHALL provide the Response in English
5. WHEN translating technical terms, THE Sahayak_System SHALL preserve accuracy while using locally understood vocabulary

### Requirement 5: Text-to-Speech Output

**User Story:** As a User who cannot read, I want to hear the answer spoken aloud in my language, so that I can understand the information without reading text.

#### Acceptance Criteria

1. WHEN a Response is generated, THE TTS_Service SHALL convert it to natural-sounding speech in the Query language
2. WHEN generating speech, THE TTS_Service SHALL use appropriate pronunciation for Local_Language terms
3. WHEN the Response is long, THE TTS_Service SHALL maintain consistent audio quality throughout
4. THE Voice_Assistant SHALL deliver audio output within 5 seconds of Query completion
5. THE TTS_Service SHALL generate audio in a format compatible with common mobile devices

### Requirement 6: Snap & Explain Document Processing

**User Story:** As a User who receives physical government documents, I want to take a photo and get a simple explanation, so that I understand what the document says and what actions I need to take.

#### Acceptance Criteria

1. WHEN a User uploads an image of a Document, THE OCR_Service SHALL extract text from the image
2. WHEN text is extracted, THE Snap_Explain_Module SHALL detect the language of the Document
3. WHEN the Document language is detected, THE Snap_Explain_Module SHALL translate it to the User's preferred Local_Language
4. WHEN a Document is processed, THE Snap_Explain_Module SHALL generate a simplified summary in plain language
5. WHEN a Document contains application instructions, THE Snap_Explain_Module SHALL extract and present them as Action_Steps
6. IF the image quality is too poor for OCR, THEN THE Snap_Explain_Module SHALL request the User to upload a clearer image

### Requirement 7: Knowledge Base Management

**User Story:** As a system administrator, I want to maintain an up-to-date knowledge base of government schemes, so that users receive current and accurate information.

#### Acceptance Criteria

1. THE Knowledge_Base SHALL include Documents from MyGov.in, Assam.gov.in, and official scheme PDFs
2. WHEN a Document is added, THE Sahayak_System SHALL chunk it into segments of 500-1000 tokens
3. WHEN Documents are chunked, THE Sahayak_System SHALL generate embeddings for each Chunk
4. WHEN embeddings are generated, THE Sahayak_System SHALL store them in the Vector_Database with metadata
5. THE Knowledge_Base SHALL include schemes such as Orunodoi, Ration Card, PMAY, and other Assam-specific programs
6. WHEN Documents are updated, THE Sahayak_System SHALL re-index the affected content within 24 hours

### Requirement 8: Performance and Latency

**User Story:** As a User with limited patience and connectivity, I want to receive answers quickly, so that I can get the information I need without long waits.

#### Acceptance Criteria

1. THE Voice_Assistant SHALL provide complete voice-to-voice responses within 5 seconds for 90% of queries
2. WHEN processing a Query, THE ASR_Service SHALL complete speech-to-text conversion within 3 seconds
3. WHEN searching the Vector_Database, THE RAG_Pipeline SHALL retrieve relevant Context within 1 second
4. WHEN generating a Response, THE LLM_Engine SHALL complete processing within 2 seconds
5. THE Snap_Explain_Module SHALL process uploaded images and provide summaries within 8 seconds

### Requirement 9: Accessibility and Interface Options

**User Story:** As a User with varying levels of technology access, I want multiple ways to interact with the system, so that I can use whatever communication channel is available to me.

#### Acceptance Criteria

1. WHERE WhatsApp is available, THE Sahayak_System SHALL provide voice interaction through WhatsApp Business API
2. WHERE phone service is available, THE Sahayak_System SHALL provide voice interaction through Twilio Voice
3. WHERE internet access is available, THE Sahayak_System SHALL provide a web interface for voice and text interaction
4. WHEN using the web interface, THE Sahayak_System SHALL support both voice input and text input
5. THE Sahayak_System SHALL function on basic smartphones with Android 8.0 or higher

### Requirement 10: Low Bandwidth Optimization

**User Story:** As a User in a rural area with poor internet connectivity, I want the system to work with limited bandwidth, so that I can access information even with slow connections.

#### Acceptance Criteria

1. WHEN network bandwidth is below 100 kbps, THE Sahayak_System SHALL compress audio to maintain functionality
2. WHEN transmitting voice data, THE Sahayak_System SHALL use efficient audio codecs to minimize data transfer
3. WHEN network latency exceeds 2 seconds, THE Sahayak_System SHALL display a waiting indicator to the User
4. THE Voice_Assistant SHALL cache frequently requested scheme information to reduce repeated data transfer
5. WHEN connectivity is intermittent, THE Sahayak_System SHALL queue requests and process them when connection is restored

### Requirement 11: Security and Privacy

**User Story:** As a User concerned about privacy, I want my voice data to be handled securely, so that my personal information and queries remain confidential.

#### Acceptance Criteria

1. WHEN processing voice input, THE Sahayak_System SHALL not store audio recordings permanently
2. WHEN transmitting data, THE Sahayak_System SHALL use encrypted connections for all API communications
3. WHEN logging interactions, THE Sahayak_System SHALL anonymize User identifiers
4. THE Sahayak_System SHALL not collect personally identifiable information without explicit User consent
5. WHEN a session ends, THE Sahayak_System SHALL delete temporary audio and text data within 1 hour

### Requirement 12: Error Handling and Fallback

**User Story:** As a User who may encounter system errors, I want clear feedback when something goes wrong, so that I know what to do next.

#### Acceptance Criteria

1. IF the ASR_Service fails to process audio, THEN THE Sahayak_System SHALL inform the User in their Local_Language and request retry
2. IF the Vector_Database is unavailable, THEN THE Sahayak_System SHALL inform the User that the service is temporarily unavailable
3. IF the LLM_Engine fails to generate a Response, THEN THE Sahayak_System SHALL provide a fallback message with contact information
4. IF the TTS_Service fails, THEN THE Sahayak_System SHALL display the Response as text
5. WHEN any component fails, THE Sahayak_System SHALL log the error with sufficient detail for debugging

### Requirement 13: Scalability

**User Story:** As a system operator, I want the system to handle increasing user load, so that it can scale from Assam to other states across India.

#### Acceptance Criteria

1. THE Sahayak_System SHALL support at least 100 concurrent Users during the hackathon demo
2. WHEN User load increases, THE Sahayak_System SHALL scale horizontally by adding processing instances
3. THE RAG_Pipeline SHALL maintain response times under 5 seconds with up to 1000 concurrent queries
4. THE Vector_Database SHALL support indexing of at least 10,000 Document chunks
5. WHEN expanding to new states, THE Sahayak_System SHALL support adding new Local_Languages through configuration

### Requirement 14: Content Accuracy and Source Attribution

**User Story:** As a User relying on government scheme information, I want accurate information with clear sources, so that I can trust the guidance provided.

#### Acceptance Criteria

1. WHEN providing scheme information, THE Response SHALL include the source Document name
2. WHEN eligibility criteria are stated, THE Response SHALL cite the official scheme guidelines
3. THE LLM_Engine SHALL not generate information that contradicts the retrieved Context
4. WHEN uncertainty exists, THE Response SHALL indicate that the User should verify with official sources
5. THE Knowledge_Base SHALL include metadata indicating the last update date for each Document

### Requirement 15: Hackathon Demo Capabilities

**User Story:** As a hackathon participant, I want to demonstrate core functionality within time and resource constraints, so that judges can evaluate the system's potential.

#### Acceptance Criteria

1. THE Sahayak_System SHALL demonstrate voice-to-voice interaction in at least Assamese and Hindi
2. THE Sahayak_System SHALL demonstrate Snap & Explain functionality with sample government documents
3. THE Knowledge_Base SHALL include at least 20 government schemes relevant to Assam
4. THE Sahayak_System SHALL provide end-to-end responses for at least 10 common civic queries
5. THE demo SHALL run on cloud infrastructure with API cost under $50 for the hackathon duration

## Non-Functional Requirements

### NFR-1: Response Quality
- Responses SHALL be grammatically correct in the target Local_Language
- Responses SHALL use vocabulary appropriate for users with basic literacy levels
- Responses SHALL be concise while providing complete information

### NFR-2: Reliability
- THE Sahayak_System SHALL maintain 95% uptime during the hackathon demo period
- Component failures SHALL not cause complete system unavailability

### NFR-3: Maintainability
- Code SHALL be modular to allow independent updates to ASR, LLM, and TTS components
- THE Knowledge_Base SHALL support updates without system downtime

### NFR-4: Cost Efficiency
- API costs SHALL be optimized through caching and lightweight model selection
- THE LLM_Engine SHALL use context windows limited to 4000 tokens to reduce costs

### NFR-5: Usability
- Voice interaction SHALL require no more than 2 steps: speak query, receive answer
- Error messages SHALL be clear and actionable in the User's Local_Language

## Constraints

1. **Hackathon Timeline**: Complete system must be demonstrable within hackathon duration (typically 24-48 hours)
2. **Dataset Access**: Limited to publicly available government documents and scheme PDFs
3. **API Costs**: Total API usage must remain within hackathon budget constraints (approximately $50)
4. **Model Selection**: Must use accessible LLMs (Llama 3, Gemini 1.5 Flash) and ASR services (Bhashini, Whisper)
5. **Infrastructure**: Must deploy on standard cloud platforms (AWS, GCP, or Azure)
6. **Language Support**: Initial focus on Assamese, Bodo, and Hindi with English fallback
7. **Testing**: Limited real-world user testing due to hackathon timeframe

## Assumptions

1. Target users have access to basic smartphones with internet connectivity
2. Government scheme PDFs are publicly accessible and can be legally scraped
3. Bhashini API or equivalent ASR service supports Assamese and Bodo languages
4. Users are willing to use voice interaction for civic queries
5. WhatsApp Business API or Twilio Voice can be accessed for demo purposes
6. Vector database services (Pinecone or Milvus) are available within budget
7. Sample government documents can be obtained for Snap & Explain demonstration

## Future Enhancements

1. **SMS Fallback**: Support for users without smartphone access through SMS-based interaction
2. **Offline Edge Deployment**: Local processing capability for areas with no internet connectivity
3. **State Database Integration**: Direct integration with government databases for real-time scheme status
4. **User Profiles**: Personalized recommendations based on user demographics and location
5. **Multi-State Expansion**: Support for additional Indian states and regional languages
6. **Voice Authentication**: Secure user identification for personalized services
7. **Feedback Loop**: User feedback mechanism to improve response quality
8. **Analytics Dashboard**: Usage analytics for government agencies to understand citizen needs
9. **Scheme Application Integration**: Direct application submission through the system
10. **Community Features**: Peer-to-peer information sharing among rural users
