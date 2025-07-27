# Pandoc Template System Implementation

## Template Architecture

### Base HTML5 Template Structure
```html
<!DOCTYPE html>
<html lang="$lang$" dir="$dir$">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes">
  <meta name="generator" content="Memoire AI Book Generator">
  
  $if(title)$
  <title>$title$$if(subtitle)$ - $subtitle$$endif$</title>
  $endif$
  
  $if(author)$
  <meta name="author" content="$for(author)$$author$$sep$, $endfor$">
  $endif$
  
  $if(description)$
  <meta name="description" content="$description$">
  $endif$
  
  $if(keywords)$
  <meta name="keywords" content="$for(keywords)$$keywords$$sep$, $endfor$">
  $endif$
  
  <!-- Open Graph / Facebook -->
  <meta property="og:type" content="book">
  <meta property="og:title" content="$title$">
  <meta property="og:description" content="$description$">
  
  <!-- CSS Styles -->
  <link rel="stylesheet" href="$css-framework$">
  <link rel="stylesheet" href="$custom-css$">
  
  $for(css)$
  <link rel="stylesheet" href="$css$">
  $endfor$
  
  <style>
    $styles.css()$
  </style>
</head>

<body class="$body-class$">
  $if(include-before)$
  $for(include-before)$
  $include-before$
  $endfor$
  $endif$

  <div id="book-container" class="container">
    $if(title)$
    <header id="book-header">
      <div class="title-section">
        <h1 class="book-title">$title$</h1>
        $if(subtitle)$
        <p class="book-subtitle">$subtitle$</p>
        $endif$
        $if(author)$
        <p class="book-author">by $for(author)$$author$$sep$, $endfor$</p>
        $endif$
        $if(date)$
        <p class="book-date">$date$</p>
        $endif$
      </div>
    </header>
    $endif$

    $if(toc)$
    <nav id="table-of-contents">
      <h2>Table of Contents</h2>
      $table-of-contents$
    </nav>
    $endif$

    <main id="book-content">
      $body$
    </main>

    $if(include-after)$
    $for(include-after)$
    $include-after$
    $endfor$
    $endif$
  </div>

  <!-- JavaScript -->
  $for(js)$
  <script src="$js$"></script>
  $endfor$
  
  <script>
    $scripts.js()$
  </script>
</body>
</html>
```

### CSS Framework Integration
```css
/* Base Book Styling */
:root {
  --primary-color: #2c3e50;
  --secondary-color: #34495e;
  --text-color: #2c3e50;
  --background-color: #ffffff;
  --accent-color: #3498db;
  --border-color: #ecf0f1;
  
  --font-family-serif: 'Georgia', 'Times New Roman', serif;
  --font-family-sans: 'Helvetica Neue', 'Arial', sans-serif;
  --font-family-mono: 'Courier New', monospace;
  
  --line-height-base: 1.6;
  --font-size-base: 18px;
  --max-width: 800px;
}

/* Typography */
body {
  font-family: var(--font-family-serif);
  font-size: var(--font-size-base);
  line-height: var(--line-height-base);
  color: var(--text-color);
  background-color: var(--background-color);
  margin: 0;
  padding: 0;
}

.container {
  max-width: var(--max-width);
  margin: 0 auto;
  padding: 2rem;
}

/* Book Header */
#book-header {
  text-align: center;
  margin-bottom: 3rem;
  padding-bottom: 2rem;
  border-bottom: 2px solid var(--border-color);
}

.book-title {
  font-size: 2.5rem;
  font-weight: bold;
  margin-bottom: 0.5rem;
  color: var(--primary-color);
}

.book-subtitle {
  font-size: 1.25rem;
  font-style: italic;
  margin-bottom: 1rem;
  color: var(--secondary-color);
}

.book-author {
  font-size: 1.1rem;
  margin-bottom: 0.5rem;
}

/* Table of Contents */
#table-of-contents {
  background-color: #f8f9fa;
  padding: 1.5rem;
  border-radius: 8px;
  margin-bottom: 3rem;
}

#table-of-contents h2 {
  margin-top: 0;
  color: var(--primary-color);
}

#table-of-contents ul {
  list-style: none;
  padding-left: 0;
}

#table-of-contents li {
  margin-bottom: 0.5rem;
}

#table-of-contents a {
  color: var(--accent-color);
  text-decoration: none;
  padding: 0.25rem 0;
  display: block;
}

#table-of-contents a:hover {
  text-decoration: underline;
}

/* Chapter Styling */
.chapter {
  margin-bottom: 4rem;
  page-break-before: always;
}

.chapter-title {
  font-size: 2rem;
  color: var(--primary-color);
  margin-bottom: 1.5rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border-color);
}

.chapter-number {
  font-size: 1rem;
  color: var(--secondary-color);
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 0.5rem;
}

/* Paragraph and Text */
p {
  margin-bottom: 1.2rem;
  text-align: justify;
}

blockquote {
  margin: 1.5rem 0;
  padding: 1rem 1.5rem;
  border-left: 4px solid var(--accent-color);
  background-color: #f8f9fa;
  font-style: italic;
}

/* Responsive Design */
@media (max-width: 768px) {
  .container {
    padding: 1rem;
  }
  
  .book-title {
    font-size: 2rem;
  }
  
  .chapter-title {
    font-size: 1.5rem;
  }
  
  :root {
    --font-size-base: 16px;
  }
}

/* Print Styles */
@media print {
  body {
    font-size: 12pt;
    line-height: 1.4;
  }
  
  .chapter {
    page-break-before: always;
  }
  
  #table-of-contents {
    page-break-after: always;
  }
  
  a {
    color: inherit;
    text-decoration: none;
  }
}
```

## Pandoc Conversion Service

### Conversion Engine Implementation
```python
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import json

class PandocConverter:
    def __init__(self, templates_dir: str = "./templates"):
        self.templates_dir = Path(templates_dir)
        self.ensure_templates_exist()
    
    def ensure_templates_exist(self):
        """Ensure template directory and default templates exist"""
        self.templates_dir.mkdir(exist_ok=True)
        
        # Create default template if it doesn't exist
        default_template = self.templates_dir / "book.html5"
        if not default_template.exists():
            self.create_default_template(default_template)
    
    async def convert_book_to_html(self, 
                                  chapters: List[Dict[str, Any]], 
                                  book_metadata: Dict[str, Any],
                                  template_name: str = "book.html5",
                                  custom_css: Optional[str] = None) -> str:
        """Convert markdown chapters to complete HTML book"""
        
        # Prepare markdown content
        full_markdown = self._prepare_full_markdown(chapters, book_metadata)
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as md_file:
            md_file.write(full_markdown)
            md_path = md_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as html_file:
            html_path = html_file.name
        
        try:
            # Build pandoc command
            cmd = self._build_pandoc_command(
                input_file=md_path,
                output_file=html_path,
                template_name=template_name,
                metadata=book_metadata,
                custom_css=custom_css
            )
            
            # Execute conversion
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Read generated HTML
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            return html_content
            
        except subprocess.CalledProcessError as e:
            raise ConversionError(f"Pandoc conversion failed: {e.stderr}")
        finally:
            # Cleanup temporary files
            os.unlink(md_path)
            os.unlink(html_path)
    
    def _prepare_full_markdown(self, chapters: List[Dict[str, Any]], 
                              book_metadata: Dict[str, Any]) -> str:
        """Combine all chapters into a single markdown document"""
        
        # Start with YAML frontmatter
        frontmatter = self._generate_frontmatter(book_metadata)
        
        # Add chapters
        markdown_parts = [frontmatter]
        
        for i, chapter in enumerate(chapters, 1):
            chapter_md = self._format_chapter_markdown(chapter, i)
            markdown_parts.append(chapter_md)
        
        return "\n\n".join(markdown_parts)
    
    def _generate_frontmatter(self, metadata: Dict[str, Any]) -> str:
        """Generate YAML frontmatter for the book"""
        
        frontmatter_data = {
            "title": metadata.get("title", "Untitled Book"),
            "subtitle": metadata.get("subtitle"),
            "author": metadata.get("author", ["Unknown Author"]),
            "date": metadata.get("created_at", "").split("T")[0] if metadata.get("created_at") else None,
            "description": metadata.get("description"),
            "lang": "en",
            "toc": True,
            "toc-depth": 2,
            "css-framework": "styles/book.css",
            "custom-css": "styles/custom.css",
            "body-class": "book-layout"
        }
        
        # Remove None values
        frontmatter_data = {k: v for k, v in frontmatter_data.items() if v is not None}
        
        yaml_content = yaml.dump(frontmatter_data, default_flow_style=False)
        return f"---\n{yaml_content}---"
    
    def _format_chapter_markdown(self, chapter: Dict[str, Any], chapter_num: int) -> str:
        """Format a single chapter as markdown"""
        
        title = chapter.get("title", f"Chapter {chapter_num}")
        content = chapter.get("content", {}).get("markdown_text", "")
        
        # Add chapter header with proper ID for TOC
        chapter_id = f"chapter-{chapter_num}"
        chapter_header = f"# {title} {{#{chapter_id}}}\n\n"
        
        return chapter_header + content
    
    def _build_pandoc_command(self, 
                             input_file: str,
                             output_file: str,
                             template_name: str,
                             metadata: Dict[str, Any],
                             custom_css: Optional[str] = None) -> List[str]:
        """Build the pandoc command with all options"""
        
        template_path = self.templates_dir / template_name
        
        cmd = [
            "pandoc",
            input_file,
            "-o", output_file,
            "--template", str(template_path),
            "--standalone",
            "--toc",
            "--toc-depth=2",
            "--section-divs",
            "--html-q-tags",
            "--from", "markdown+smart+yaml_metadata_block",
            "--to", "html5"
        ]
        
        # Add CSS files
        css_dir = Path("./templates/styles")
        if (css_dir / "book.css").exists():
            cmd.extend(["--css", str(css_dir / "book.css")])
        
        if custom_css and Path(custom_css).exists():
            cmd.extend(["--css", custom_css])
        
        # Add metadata variables
        for key, value in metadata.items():
            if value:
                cmd.extend(["-V", f"{key}={value}"])
        
        return cmd

class ConversionError(Exception):
    pass
```

### Template Management System
```python
class TemplateManager:
    def __init__(self, templates_dir: str):
        self.templates_dir = Path(templates_dir)
        self.available_templates = self._discover_templates()
    
    def _discover_templates(self) -> Dict[str, Dict[str, Any]]:
        """Discover available templates and their metadata"""
        
        templates = {}
        
        for template_file in self.templates_dir.glob("*.html5"):
            template_name = template_file.stem
            metadata = self._extract_template_metadata(template_file)
            
            templates[template_name] = {
                "file": template_file,
                "name": template_name,
                "description": metadata.get("description", ""),
                "author": metadata.get("author", ""),
                "version": metadata.get("version", "1.0"),
                "supports": metadata.get("supports", []),
                "css_files": self._find_template_css(template_name)
            }
        
        return templates
    
    def get_template_config(self, template_name: str) -> Dict[str, Any]:
        """Get configuration for a specific template"""
        
        if template_name not in self.available_templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        return self.available_templates[template_name]
    
    def create_custom_template(self, 
                              base_template: str,
                              customizations: Dict[str, Any],
                              template_name: str) -> str:
        """Create a custom template based on existing template"""
        
        base_config = self.get_template_config(base_template)
        base_content = base_config["file"].read_text()
        
        # Apply customizations
        customized_content = self._apply_customizations(base_content, customizations)
        
        # Save custom template
        custom_template_path = self.templates_dir / f"{template_name}.html5"
        custom_template_path.write_text(customized_content)
        
        # Refresh available templates
        self.available_templates = self._discover_templates()
        
        return template_name
    
    def _apply_customizations(self, template_content: str, 
                            customizations: Dict[str, Any]) -> str:
        """Apply customizations to template content"""
        
        # Replace color variables
        if "colors" in customizations:
            for color_var, color_value in customizations["colors"].items():
                template_content = template_content.replace(
                    f"var(--{color_var})", color_value
                )
        
        # Replace font variables
        if "fonts" in customizations:
            for font_var, font_value in customizations["fonts"].items():
                template_content = template_content.replace(
                    f"var(--{font_var})", font_value
                )
        
        # Add custom CSS
        if "custom_css" in customizations:
            css_injection = f"<style>\n{customizations['custom_css']}\n</style>"
            template_content = template_content.replace("</head>", f"{css_injection}\n</head>")
        
        return template_content
```

### Conversion API Service
```python
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any

router = APIRouter(tags=["HTML Conversion"])

class ConversionRequest(BaseModel):
    book_id: str
    book_version: str = "latest"
    template_name: str = "book.html5"
    custom_css: Optional[str] = None
    conversion_options: Dict[str, Any] = {}

class ConversionResponse(BaseModel):
    conversion_id: str
    status: str
    download_url: Optional[str] = None
    error_message: Optional[str] = None

@router.post("/convert-to-html", response_model=ConversionResponse)
async def convert_book_to_html(
    request: ConversionRequest,
    background_tasks: BackgroundTasks
) -> ConversionResponse:
    """Convert a book to HTML format using Pandoc"""
    
    try:
        # Get book and chapters
        book_metadata = await storage_service.get_book_metadata(request.book_id)
        chapters = await storage_service.get_book_chapters(
            request.book_id, 
            request.book_version
        )
        
        if not chapters:
            raise HTTPException(status_code=404, detail="No chapters found for book")
        
        # Start conversion in background
        conversion_id = f"conv_{request.book_id}_{int(time.time())}"
        
        background_tasks.add_task(
            execute_conversion,
            conversion_id,
            chapters,
            book_metadata,
            request.template_name,
            request.custom_css,
            request.conversion_options
        )
        
        return ConversionResponse(
            conversion_id=conversion_id,
            status="processing"
        )
        
    except Exception as e:
        logger.error(f"Conversion initiation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def execute_conversion(
    conversion_id: str,
    chapters: List[Dict],
    book_metadata: Dict,
    template_name: str,
    custom_css: Optional[str],
    options: Dict[str, Any]
):
    """Execute the actual conversion process"""
    
    try:
        converter = PandocConverter()
        
        # Convert to HTML
        html_content = await converter.convert_book_to_html(
            chapters=chapters,
            book_metadata=book_metadata,
            template_name=template_name,
            custom_css=custom_css
        )
        
        # Store the result
        output_path = f"./outputs/{conversion_id}.html"
        Path(output_path).parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Update conversion status
        await storage_service.update_conversion_status(
            conversion_id, 
            "completed", 
            output_path
        )
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        await storage_service.update_conversion_status(
            conversion_id, 
            "failed", 
            error_message=str(e)
        )

@router.get("/conversion-status/{conversion_id}")
async def get_conversion_status(conversion_id: str) -> ConversionResponse:
    """Get the status of a conversion process"""
    
    status_record = await storage_service.get_conversion_status(conversion_id)
    
    if not status_record:
        raise HTTPException(status_code=404, detail="Conversion not found")
    
    download_url = None
    if status_record["status"] == "completed" and status_record.get("output_path"):
        download_url = f"/download/{conversion_id}"
    
    return ConversionResponse(
        conversion_id=conversion_id,
        status=status_record["status"],
        download_url=download_url,
        error_message=status_record.get("error_message")
    )

@router.get("/templates")
async def list_available_templates() -> Dict[str, Any]:
    """List all available conversion templates"""
    
    template_manager = TemplateManager("./templates")
    
    return {
        "templates": template_manager.available_templates,
        "total_count": len(template_manager.available_templates)
    }
```

This Pandoc template system provides a robust, customizable HTML conversion pipeline that can generate high-quality web-ready books from markdown chapters while maintaining professional styling and responsive design.