# Phase 4: Output Management & Publishing (MVP)
# AgentLang program for simplified content management and HTML publishing

# MVP Constraints from answers:
# - Firestore storage with metadata tags (user > book > chapter)
# - Database-based version tracking (no Git)
# - Sequential chapter generation (one writer agent at a time)
# - No collaborative editing, approval workflows, or edit history
# - HTML output only with Pandoc conversion
# - Chat interface for follow-up questions
# - Manual publishing via UI button
# - Simple UI marketplace (no external integrations)

# Input: AI processing implementation from Phase 3
firestore_content_schema = breakdown:tree(ai_implementation) → .md
markdown_storage_design = act:draft(firestore_content_schema) → .md
version_tracking_system = breakdown:tree(markdown_storage_design) → .md
sequential_chapter_workflow = act:draft(version_tracking_system) → .md
markdown_validation_service = breakdown:tree(sequential_chapter_workflow) → .md
chat_interface_design = act:draft(markdown_validation_service) → .md
html_conversion_pipeline = breakdown:tree(chat_interface_design) → .md
pandoc_template_system = act:draft(html_conversion_pipeline) → .md
ui_marketplace_design = breakdown:tree(pandoc_template_system) → .md
manual_publishing_workflow = act:draft(ui_marketplace_design) → .md
output_implementation = act:implement(manual_publishing_workflow)

END PROGRAM