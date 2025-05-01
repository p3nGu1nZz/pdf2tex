"""Module for handling reference extraction and BibTeX generation from PDF content."""

import re
from .utils import Utils

class ReferenceExtractor:
    """Extracts and formats references from PDF text content into BibTeX format."""

    @staticmethod
    def identify_reference_section(text):
        """
        Determine if a text block is likely a reference section.
        Returns True if this appears to be a reference/bibliography section.
        """
        if not text:
            return False
            
        # Check common section headers that indicate references
        reference_headers = [
            r"^\s*references\s*$",
            r"^\s*bibliography\s*$", 
            r"^\s*references cited\s*$",
            r"^\s*works cited\s*$",
            r"^\s*cited works\s*$",
            r"^\s*citations\s*$"
        ]
        
        first_line = text.strip().split('\n')[0].lower()
        
        for pattern in reference_headers:
            if re.match(pattern, first_line, re.IGNORECASE):
                return True
                
        # Look for characteristic citation patterns at the beginning of lines
        citation_patterns = [
            r"^\s*\[\d+\]",                        # [1] Format
            r"^\s*\[\w+\d+\w*\]",                  # [ABC12] Format
            r"^\s*\d+\.\s",                        # 1. Format
            r"^\s*[A-Z][a-z]+,\s*[A-Z]\.",         # Lastname, F. Format
            r"^\s*[A-Z][a-z]+\s+et\s+al\.\s*,",    # Lastname et al.
        ]
        
        lines = text.strip().split('\n')
        citation_line_count = 0
        for line in lines[:min(10, len(lines))]:  # Check first 10 lines
            for pattern in citation_patterns:
                if re.match(pattern, line):
                    citation_line_count += 1
                    break
                    
        # If many lines match citation patterns, likely a reference section
        if citation_line_count >= 3 or (len(lines) <= 10 and citation_line_count >= 2):
            return True
            
        return False

    @staticmethod
    def extract_bibtex_entries(text):
        """
        Extract references from text and convert them to BibTeX entries.
        Returns a list of BibTeX entry strings.
        """
        entries = []
        
        # First, try to identify citation blocks by common patterns
        # These patterns look for citation keys like [1], [ABC12], or Author et al.
        citation_patterns = [
            # [Key] Citation pattern - matches [ABC98] Smith et al. "Title"...
            r'\[([^\]]+)\](.*?)(?=\[\w+\]|\Z)',
            
            # Numbered pattern - matches 1. Author et al. ...
            r'(\d+\.)(.*?)(?=\d+\.|\Z)',
            
            # Author (Year) pattern - matches "Smith et al. (2020)"
            r'([A-Z][a-z]+(?:\s+et\s+al\.)?)\s*\((\d{4}[a-z]?)\)(.*?)(?=[A-Z][a-z]+\s+\(\d{4}\)|\Z)',
            
            # Author Year pattern - no parentheses, separated by spacing
            r'([A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+|\s+et\s+al\.)?)\s+(\d{4}[a-z]?)(.*?)(?=[A-Z][a-z]+\s+\d{4}|\Z)',
        ]
        
        # Try each pattern until we find something that works
        matches = []
        for pattern in citation_patterns:
            matches = list(re.finditer(pattern, text, re.DOTALL))
            if len(matches) >= 2:  # Found multiple matches with this pattern
                break
                
        # If no structured pattern was found, try simple line-by-line parsing
        if not matches:
            # Split by common line breaks in references
            lines = re.split(r'\n(?:\s*\n)+', text.strip())
            for i, line in enumerate(lines):
                if not line.strip():
                    continue
                    
                entry_id = f"ref{i+1}"
                entry = {
                    'key': entry_id,
                    'title': '',
                    'author': '',
                    'year': '',
                    'journal': '',
                    'raw': line.strip()
                }
                
                # Try to extract year if present (4 digits that look like a year)
                year_match = re.search(r'\b(19|20)\d{2}\b', line)
                if year_match:
                    entry['year'] = year_match.group(0)
                    
                # Try to extract title if in quotes
                title_match = re.search(r'"([^"]+)"', line)
                if title_match:
                    entry['title'] = title_match.group(1)
                elif title_match := re.search(r'"([^"]+)"', line):
                    entry['title'] = title_match.group(1)
                    
                entries.append(ReferenceExtractor.format_bibtex(entry))
        else:
            # Process structured matches
            for i, match in enumerate(matches):
                if len(match.groups()) >= 2:
                    key = match.group(1).strip()
                    content = match.group(2).strip()
                    
                    # Create citation key - clean up non-alphanumeric chars
                    citation_key = re.sub(r'[^\w]', '', key)
                    if not citation_key:
                        citation_key = f"ref{i+1}"
                    
                    entry = {
                        'key': citation_key,
                        'title': '',
                        'author': '',
                        'year': '',
                        'journal': '',
                        'raw': content
                    }
                    
                    # Try to extract year
                    year_match = re.search(r'\b(19|20)\d{2}\b', content)
                    if year_match:
                        entry['year'] = year_match.group(0)
                        
                    # Try to extract title from quotes
                    title_match = re.search(r'"([^"]+)"', content)
                    if title_match:
                        entry['title'] = title_match.group(1)
                    elif title_match := re.search(r'"([^"]+)"', content):
                        entry['title'] = title_match.group(1)
                    
                    # Try to extract author information
                    # This is a simplified approach - author detection is complex
                    author_match = re.match(r'([^,\.]+)', content)
                    if author_match:
                        entry['author'] = author_match.group(1).strip()
                        
                    entries.append(ReferenceExtractor.format_bibtex(entry))
                    
        # If we still have no entries but have text, create a fallback entry
        if not entries and text.strip():
            entries.append(f"@misc{{unknown_references,\n  note = {{{Utils.escape_special_chars(text.strip())}}}\n}}\n")
            
        return entries

    @staticmethod
    def format_bibtex(entry):
        """
        Format a reference entry dictionary into a BibTeX string.
        """
        # If we have structured data, create a more specific BibTeX entry
        if entry.get('title') or entry.get('author') or entry.get('year'):
            fields = []
            
            if entry.get('author'):
                fields.append(f"  author = {{{Utils.escape_special_chars(entry['author'])}}}")
                
            if entry.get('title'):
                fields.append(f"  title = {{{Utils.escape_special_chars(entry['title'])}}}")
                
            if entry.get('year'):
                fields.append(f"  year = {{{entry['year']}}}")
                
            if entry.get('journal'):
                fields.append(f"  journal = {{{Utils.escape_special_chars(entry['journal'])}}}")
                
            # Always include the raw text as a note for completeness
            if entry.get('raw'):
                fields.append(f"  note = {{{Utils.escape_special_chars(entry['raw'])}}}")
                
            # If we don't have core fields, default to misc
            entry_type = "article" if entry.get('journal') else "misc"
            
            return f"@{entry_type}{{{entry['key']},\n{',\n'.join(fields)}\n}}\n"
        else:
            # Fallback to simple misc entry with everything in the note field
            return f"@misc{{{entry['key']},\n  note = {{{Utils.escape_special_chars(entry['raw'])}}}\n}}\n"

    @staticmethod
    def merge_bibtex_entries(entries):
        """
        Merge and clean a list of BibTeX entries.
        Returns a single formatted BibTeX string.
        """
        # Basic deduplication by comparing raw text
        unique_entries = []
        seen_content = set()
        
        for entry in entries:
            # Skip duplicates or empty entries
            if not entry or entry in seen_content:
                continue
                
            unique_entries.append(entry)
            seen_content.add(entry)
            
        # Join all entries with an empty line between them
        merged = "\n".join(unique_entries)
        
        # Add BibTeX header comment
        header = "% BibTeX references extracted from PDF by pdf2tex\n\n"
        
        return header + merged