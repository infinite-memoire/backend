"""
HTML Conversion Service

Converts book content to HTML format for publishing and preview generation.
"""

from typing import Dict, List, Any
from pathlib import Path
import jinja2

from app.utils.logging import get_logger

logger = get_logger("html_conversion")


class HTMLConversionService:
    """Service for converting book content to HTML"""
    
    def __init__(self, template_dir: str = "./templates"):
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(exist_ok=True)
        
        # Initialize Jinja2 environment
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
    
    async def convert_book_to_html(self, chapters: List[Dict[str, Any]], 
                                 book_metadata: Dict[str, Any],
                                 template_name: str = "book.html5") -> str:
        """Convert book chapters to HTML using specified template"""
        
        try:
            # Load template
            template = self.jinja_env.get_template(template_name)
            
            # Prepare context data
            context = {
                "book": book_metadata,
                "chapters": chapters,
                "metadata": {
                    "title": book_metadata.get("title", "Untitled"),
                    "author": book_metadata.get("author_name", "Unknown Author"),
                    "description": book_metadata.get("description", ""),
                    "language": book_metadata.get("language", "en"),
                    "publication_date": book_metadata.get("publication_date"),
                    "cover_image_url": book_metadata.get("cover_image_url"),
                    "estimated_reading_time": book_metadata.get("estimated_reading_time", 0)
                },
                "settings": {
                    "allow_comments": book_metadata.get("allow_comments", True),
                    "allow_downloads": book_metadata.get("allow_downloads", True),
                    "copyright_notice": book_metadata.get("copyright_notice", ""),
                    "license_type": book_metadata.get("license_type", "all_rights_reserved")
                }
            }
            
            # Render HTML
            html_content = template.render(**context)
            
            logger.info(f"Successfully converted book to HTML using template: {template_name}")
            return html_content
            
        except jinja2.TemplateNotFound:
            # Use default template if specified template not found
            logger.warning(f"Template {template_name} not found, using default template")
            return await self._render_default_template(chapters, book_metadata)
        
        except Exception as e:
            logger.error(f"Failed to convert book to HTML: {str(e)}")
            raise HTMLConversionError(f"HTML conversion failed: {str(e)}")
    
    async def _render_default_template(self, chapters: List[Dict[str, Any]], 
                                     book_metadata: Dict[str, Any]) -> str:
        """Render using default HTML template when custom template is not available"""
        
        # Create default HTML structure
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='{}'>".format(book_metadata.get("language", "en")),
            "<head>",
            "<meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1'>",
            "<title>{}</title>".format(book_metadata.get("title", "Untitled")),
            "<style>",
            self._get_default_css(),
            "</style>",
            "</head>",
            "<body>",
            "<div class='book-container'>",
            "<header class='book-header'>",
            "<h1 class='book-title'>{}</h1>".format(book_metadata.get("title", "Untitled")),
        ]
        
        # Add subtitle if exists
        if book_metadata.get("subtitle"):
            html_parts.append("<h2 class='book-subtitle'>{}</h2>".format(book_metadata["subtitle"]))
        
        # Add author
        html_parts.append("<p class='book-author'>by {}</p>".format(
            book_metadata.get("author_name", "Unknown Author")
        ))
        
        # Add description if exists
        if book_metadata.get("description"):
            html_parts.extend([
                "<div class='book-description'>",
                "<p>{}</p>".format(book_metadata["description"]),
                "</div>"
            ])
        
        html_parts.append("</header>")
        
        # Add chapters
        html_parts.append("<main class='book-content'>")
        
        for i, chapter in enumerate(chapters, 1):
            chapter_title = chapter.get("title", f"Chapter {i}")
            chapter_content = chapter.get("content", {}).get("markdown_text", "")
            
            html_parts.extend([
                "<section class='chapter'>",
                "<h2 class='chapter-title'>{}</h2>".format(chapter_title),
                "<div class='chapter-content'>",
                self._markdown_to_html(chapter_content),
                "</div>",
                "</section>"
            ])
        
        html_parts.extend([
            "</main>",
            "</div>",
            "</body>",
            "</html>"
        ])
        
        return "\n".join(html_parts)
    
    def _get_default_css(self) -> str:
        """Get default CSS styles for HTML rendering"""
        return """
        body {
            font-family: Georgia, serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }
        
        .book-container {
            background: white;
            padding: 40px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        
        .book-header {
            text-align: center;
            margin-bottom: 40px;
            padding-bottom: 20px;
            border-bottom: 2px solid #eee;
        }
        
        .book-title {
            font-size: 2.5em;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .book-subtitle {
            font-size: 1.5em;
            color: #7f8c8d;
            font-weight: normal;
            margin-bottom: 20px;
        }
        
        .book-author {
            font-size: 1.2em;
            color: #34495e;
            font-style: italic;
        }
        
        .book-description {
            margin: 20px 0;
            font-size: 1.1em;
            color: #555;
        }
        
        .chapter {
            margin-bottom: 40px;
            page-break-before: always;
        }
        
        .chapter-title {
            font-size: 1.8em;
            color: #2c3e50;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #ddd;
        }
        
        .chapter-content {
            font-size: 1.1em;
            line-height: 1.8;
        }
        
        .chapter-content p {
            margin-bottom: 1em;
            text-align: justify;
        }
        
        @media print {
            body { margin: 0; }
            .book-container { box-shadow: none; }
        }
        """
    
    def _markdown_to_html(self, markdown_text: str) -> str:
        """Convert markdown text to HTML (basic implementation)"""
        
        if not markdown_text:
            return ""
        
        # Basic markdown to HTML conversion
        # In a real implementation, you'd use a proper markdown parser like markdown or mistune
        html_text = markdown_text
        
        # Convert paragraphs
        paragraphs = html_text.split('\n\n')
        html_paragraphs = []
        
        for para in paragraphs:
            para = para.strip()
            if para:
                # Basic formatting
                para = para.replace('**', '<strong>').replace('**', '</strong>')
                para = para.replace('*', '<em>').replace('*', '</em>')
                para = para.replace('\n', '<br>')
                html_paragraphs.append(f"<p>{para}</p>")
        
        return '\n'.join(html_paragraphs)
    
    async def create_template(self, template_name: str, template_content: str) -> None:
        """Create a new HTML template"""
        
        template_path = self.template_dir / template_name
        
        try:
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write(template_content)
            
            logger.info(f"Template created: {template_name}")
            
        except Exception as e:
            logger.error(f"Failed to create template {template_name}: {str(e)}")
            raise HTMLConversionError(f"Template creation failed: {str(e)}")
    
    async def list_templates(self) -> List[str]:
        """List available templates"""
        
        try:
            return [f.name for f in self.template_dir.glob("*.html*")]
        except Exception as e:
            logger.error(f"Failed to list templates: {str(e)}")
            return []


class HTMLConversionError(Exception):
    """Exception raised during HTML conversion"""
    pass