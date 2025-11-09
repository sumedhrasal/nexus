"""Gmail source connector."""

from typing import AsyncGenerator, Optional
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from app.core.sources.base import BaseSource
from app.core.entities import EmailEntity
from app.config import settings
import logging
import base64

logger = logging.getLogger(__name__)


class GmailSource(BaseSource):
    """Gmail source connector."""

    def __init__(self, credentials: Optional[Credentials] = None):
        """Initialize Gmail source.

        Args:
            credentials: Google OAuth2 credentials
        """
        super().__init__(name="gmail")
        self.credentials = credentials
        self.service = None

    async def fetch(
        self,
        query: str = "is:unread",
        max_results: int = 100
    ) -> AsyncGenerator[EmailEntity, None]:
        """Fetch emails from Gmail.

        Args:
            query: Gmail search query
            max_results: Maximum number of emails to fetch

        Yields:
            EmailEntity instances
        """
        if not await self.validate_config():
            logger.error("Gmail credentials not configured")
            return

        try:
            self.service = build('gmail', 'v1', credentials=self.credentials)

            # List messages
            results = self.service.users().messages().list(
                userId='me',
                q=query,
                maxResults=max_results
            ).execute()

            messages = results.get('messages', [])

            for message in messages:
                # Get full message
                msg = self.service.users().messages().get(
                    userId='me',
                    id=message['id'],
                    format='full'
                ).execute()

                entity = await self._process_message(msg)
                if entity:
                    yield entity

        except Exception as e:
            logger.error(f"Failed to fetch from Gmail: {e}")

    async def _process_message(self, msg) -> Optional[EmailEntity]:
        """Process a single email message.

        Args:
            msg: Gmail message object

        Returns:
            EmailEntity or None
        """
        try:
            # Extract headers
            headers = msg['payload']['headers']
            subject = self._get_header(headers, 'Subject')
            sender = self._get_header(headers, 'From')
            to = self._get_header(headers, 'To')
            date = self._get_header(headers, 'Date')

            # Extract body
            body = self._get_body(msg['payload'])

            # Parse recipients
            recipients = [r.strip() for r in to.split(',')] if to else []

            entity = EmailEntity(
                entity_id=msg['id'],
                entity_type="email",
                content=body,
                title=subject,
                sender=sender,
                recipients=recipients,
                subject=subject,
                thread_id=msg.get('threadId'),
                metadata={
                    "date": date,
                    "labels": msg.get('labelIds', []),
                    "snippet": msg.get('snippet')
                }
            )
            return entity

        except Exception as e:
            logger.error(f"Failed to process email {msg.get('id')}: {e}")
            return None

    def _get_header(self, headers, name: str) -> Optional[str]:
        """Get header value by name.

        Args:
            headers: List of header objects
            name: Header name

        Returns:
            Header value or None
        """
        for header in headers:
            if header['name'].lower() == name.lower():
                return header['value']
        return None

    def _get_body(self, payload) -> str:
        """Extract email body from payload.

        Args:
            payload: Message payload

        Returns:
            Email body text
        """
        body = ""

        if 'parts' in payload:
            # Multipart message
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    data = part['body'].get('data')
                    if data:
                        body += base64.urlsafe_b64decode(data).decode('utf-8')
        else:
            # Simple message
            data = payload['body'].get('data')
            if data:
                body = base64.urlsafe_b64decode(data).decode('utf-8')

        return body

    async def validate_config(self) -> bool:
        """Validate Gmail credentials.

        Returns:
            True if credentials are configured
        """
        return bool(self.credentials)
