import os
import logging
from typing import List, Dict, Optional, Any

try:
    from libzim.reader import Archive
    from libzim.search import Query, Searcher
    HAS_LIBZIM = True
except ImportError:
    HAS_LIBZIM = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

logger = logging.getLogger("zim_service")

class ZimService:
    _instance = None
    _archive = None

    def __init__(self, zim_path: str):
        if not HAS_LIBZIM:
            logger.error("libzim not installed. ZIM search disabled.")
            return
        
        if not os.path.exists(zim_path):
            logger.error(f"ZIM file not found at {zim_path}")
            return

        try:
            self._archive = Archive(zim_path)
            logger.info(f"Initialized ZIM archive from {zim_path}")
        except Exception as e:
            logger.error(f"Failed to load ZIM archive: {e}")

    @classmethod
    def get_instance(cls):
        return cls._instance

    @classmethod
    def initialize(cls, zim_path: str):
        if cls._instance is None:
            cls._instance = cls(zim_path)
        return cls._instance

    def search(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Search ZIM archive by title/keywords.
        Returns a list of dicts suitable for RAG context.
        """
        if not self._archive:
            return []

        results = []
        try:
            searcher = Searcher(self._archive)
            zim_query = Query().set_query(query)
            
            # API changes compat check
            if hasattr(searcher, 'search'):
                 execution = searcher.search(zim_query)
            else:
                 execution = searcher.execute(zim_query)
            
            # Fetch results
            if hasattr(execution, 'getResults'):
                res_objs = execution.getResults(0, limit)
            else:
                res_objs = execution.get_results(0, limit)

            if not res_objs:
                return []

            for res in res_objs:
                path = None
                title = "Unknown"

                # Extract path
                if isinstance(res, str):
                    path = res
                elif hasattr(res, 'path'):
                    path = res.path
                elif hasattr(res, 'get_path'):
                    path = res.get_path()
                elif hasattr(res, 'getPath'): # Another possible camelCase
                    path = res.getPath()

                if not path:
                    continue

                # Get Entry and Content
                try:
                    entry = self._archive.get_entry_by_path(path)
                    
                    # Extract Title
                    if hasattr(entry, 'get_title'):
                        title = entry.get_title()
                    elif hasattr(entry, 'title'):
                        title = entry.title
                    elif hasattr(entry, 'getTitle'):
                        title = entry.getTitle()

                    # Extract Content
                    item = entry.get_item()
                    raw_data = b""
                    if hasattr(item, 'content'):
                        raw_data = bytes(item.content)
                    elif hasattr(item, 'get_data'):
                        raw_data = item.get_data()
                    
                    # Parse Content (PageIndex style: Extract Structure + Summary)
                    article_data = self._process_article(raw_data)
                    
                    results.append({
                        "id": f"zim_{path}",
                        "content": f"【百科: {title}】\n目录: {' > '.join(article_data['toc']) if article_data['toc'] else '无'}\n摘要: {article_data['summary']}",
                        "metadata": {
                            "source": "zim", 
                            "title": title, 
                            "path": path,
                            "toc": article_data['toc']
                        },
                        "score": 0.8, # Static score for fallback
                        "match_type": "encyclopedia_fallback"
                    })

                except Exception as e:
                    logger.warning(f"Error reading ZIM entry {path}: {e}")
                    continue

        except Exception as e:
            logger.error(f"ZIM search failed: {e}")
        
        return results

    def _process_article(self, html_bytes: bytes, max_len: int = 1500) -> Dict[str, Any]:
        """
        PageIndex Inspired: Process article to extract TOC (Tree-like structure) and a clean summary.
        Increased max_len to capture more context.
        """
        if not html_bytes:
            return {"toc": [], "summary": ""}
        
        try:
            if HAS_BS4:
                soup = BeautifulSoup(html_bytes, 'html.parser')
                
                # Clean up
                for script in soup(["script", "style", "table"]): # Tables often contain junk/sidebar info
                    script.extract()    

                # 1. Extract TOC (H2, H3 headers)
                # In Wikipedia ZIMs, headers often have specific classes or just tags
                headers = soup.find_all(['h2', 'h3'])
                toc = []
                for h in headers:
                    h_text = h.get_text().strip()
                    # Clean typical wiki artifacts like '[编辑]'
                    h_text = h_text.replace('[编辑]', '').strip()
                    if h_text and len(h_text) < 100:
                        toc.append(h_text)
                
                # 2. Extract Summary (First meaningful paragraphs)
                paragraphs = soup.find_all('p')
                summary_parts = []
                current_len = 0
                for p in paragraphs:
                    p_text = p.get_text().strip()
                    if len(p_text) > 40: # Ignore very short snips
                        summary_parts.append(p_text)
                        current_len += len(p_text)
                    if current_len >= max_len:
                        break
                
                summary = "\n\n".join(summary_parts) # Keep paragraphs separate
                
                # Soft truncation at max_len
                if len(summary) > max_len:
                     summary = summary[:max_len] + "..."

                if not summary:
                    # Fallback to general text if no paragraphs found
                    summary = soup.get_text(separator=' ', strip=True)[:max_len] + "..."

                return {
                    "toc": toc[:15], # Limit TOC size slightly relaxed
                    "summary": summary
                }
            else:
                # Fallback without BS4
                text = html_bytes.decode('utf-8', errors='ignore')
                return {"toc": [], "summary": text[:max_len] + "..."}
        except Exception as e:
            logger.warning(f"Failed to process article structure: {e}")
            return {"toc": [], "summary": "Error processing content"}

    def _parse_html_snippet(self, html_bytes: bytes, max_len: int = 1500) -> str:
        """Deprecated: Use _process_article for more structure"""
        return self._process_article(html_bytes, max_len)['summary']
