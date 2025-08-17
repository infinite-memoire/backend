from google.cloud import firestore
from app.config.settings_config import get_settings
from app.utils.logging_utils import get_logger
import firebase_admin
from firebase_admin import credentials
from typing import Optional, List, Tuple, Any
import os

logger = get_logger(__name__)

class FirestoreService:
    def __init__(self):
        settings = get_settings()
        #
        # cred = credentials.Certificate("path/to/serviceAccountKey.json")
        # firebase_admin.initialize_app(cred)

        # Initialize Firestore client
        if settings.database.firestore_emulator_host:
            # Use emulator for development
            os.environ["FIRESTORE_EMULATOR_HOST"] = settings.database.firestore_emulator_host
            self.db = firestore.Client(project=settings.database.firestore_project_id)
            logger.info("Connected to Firestore emulator", 
                       host=settings.database.firestore_emulator_host)
        else:
            # Use production Firestore
            if settings.database.firestore_credentials_path and os.path.exists(settings.database.firestore_credentials_path):
                self.db = firestore.Client.from_service_account_json(
                    settings.database.firestore_credentials_path,
                    project=settings.database.firestore_project_id
                )
                logger.info("Connected to production Firestore with service account",
                           project=settings.database.firestore_project_id)
            else:
                # Use default credentials (ADC)
                self.db = firestore.Client(project=settings.database.firestore_project_id)
                logger.info("Connected to production Firestore with default credentials",
                           project=settings.database.firestore_project_id)
        
    async def test_connection(self):
        """Test Firestore connection"""
        try:
            # Try to access a collection to test connection
            collections = list(self.db.collections())
            logger.info("Firestore connection test successful", 
                       collections_count=len(collections))
        except Exception as e:
            logger.error("Firestore connection test failed", error=str(e))
            raise
            
    async def create_audio_record(self, audio_data: dict) -> str:
        """Create a new audio record in Firestore"""
        try:
            doc_ref = self.db.collection('audio_files').document()
            doc_ref.set(audio_data)
            logger.info("Created audio record", audio_id=doc_ref.id)
            return doc_ref.id
        except Exception as e:
            logger.error("Failed to create audio record", error=str(e))
            raise
            
    async def create_transcript_record(self, transcript_data: dict) -> str:
        """Create a new transcript record in Firestore"""
        try:
            doc_ref = self.db.collection('transcripts').document()
            doc_ref.set(transcript_data)
            logger.info("Created transcript record", transcript_id=doc_ref.id)
            return doc_ref.id
        except Exception as e:
            logger.error("Failed to create transcript record", error=str(e))
            raise
            
    async def get_audio_record(self, audio_id: str) -> Optional[dict]:
        """Get an audio record by ID"""
        try:
            doc_ref = self.db.collection('audio_files').document(audio_id)
            doc = doc_ref.get()
            if doc.exists:
                logger.debug("Retrieved audio record", audio_id=audio_id)
                return doc.to_dict()
            else:
                logger.warning("Audio record not found", audio_id=audio_id)
                return None
        except Exception as e:
            logger.error("Failed to get audio record", audio_id=audio_id, error=str(e))
            raise
            
    async def update_audio_status(self, audio_id: str, status: str):
        """Update the status of an audio record"""
        try:
            doc_ref = self.db.collection('audio_files').document(audio_id)
            doc_ref.update({'upload_status': status})
            logger.info("Updated audio status", audio_id=audio_id, status=status)
        except Exception as e:
            logger.error("Failed to update audio status", 
                        audio_id=audio_id, status=status, error=str(e))
            raise
            
    async def store_audio_content(self, audio_id: str, file_content: bytes):
        """Store audio file content in Firestore"""
        try:
            doc_ref = self.db.collection('audio_content').document(audio_id)
            doc_ref.set({
                'content': file_content,
                'size': len(file_content)
            })
            logger.info("Stored audio content", 
                       audio_id=audio_id, size_bytes=len(file_content))
        except Exception as e:
            logger.error("Failed to store audio content", 
                        audio_id=audio_id, error=str(e))
            raise
            
    async def get_transcript_record(self, transcript_id: str) -> Optional[dict]:
        """Get a transcript record by ID"""
        try:
            doc_ref = self.db.collection('transcripts').document(transcript_id)
            doc = doc_ref.get()
            if doc.exists:
                logger.debug("Retrieved transcript record", transcript_id=transcript_id)
                return doc.to_dict()
            else:
                logger.warning("Transcript record not found", transcript_id=transcript_id)
                return None
        except Exception as e:
            logger.error("Failed to get transcript record", 
                        transcript_id=transcript_id, error=str(e))
            raise
            
    async def create_document(self, collection: str, document_id: str, data: dict) -> str:
        """Create a document in a specified collection"""
        try:
            doc_ref = self.db.collection(collection).document(document_id)
            doc_ref.set(data)
            logger.info("Created document", collection=collection, document_id=document_id)
            return document_id
        except Exception as e:
            logger.error("Failed to create document", 
                        collection=collection, document_id=document_id, error=str(e))
            raise
            
    async def query_documents(self, collection: str, filters: List[Tuple[str, str, Any]], 
                            order_by: Optional[List[Tuple[str, str]]] = None,
                            limit: Optional[int] = None) -> List[dict]:
        """Query documents from a collection with filters"""
        try:
            query = self.db.collection(collection)
            
            # Apply filters
            for field, operator, value in filters:
                query = query.where(field, operator, value)
                
            # Apply ordering
            if order_by:
                for field, direction in order_by:
                    query = query.order_by(field, direction=direction)
                    
            # Apply limit
            if limit:
                query = query.limit(limit)
                
            docs = query.stream()
            results = []
            for doc in docs:
                doc_data = doc.to_dict()
                doc_data['id'] = doc.id
                results.append(doc_data)
                
            logger.debug("Queried documents", collection=collection, 
                        filters_count=len(filters), results_count=len(results))
            return results
        except Exception as e:
            logger.error("Failed to query documents", 
                        collection=collection, error=str(e))
            raise

# Global service instance
firestore_service = FirestoreService()
