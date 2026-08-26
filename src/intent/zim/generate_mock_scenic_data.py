import json
import os
import sys
import hashlib
import multiprocessing
from functools import partial

# Ensure project root is in path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import ZimService for real extraction
try:
    from src.rag.zim_service import ZimService
    from bs4 import BeautifulSoup
except ImportError:
    # Quick fix if running from within src/..
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    from rag.zim_service import ZimService
    from bs4 import BeautifulSoup

# Configuration
ZIM_PATH = os.path.join(project_root, "src/intent/data/wikipedia_zh_all_maxi_2023-09.zim")
OUTPUT_DIR = "mock_scenic_data"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Meta-information to exclude
IGNORED_HEADERS = {
    '参考文献', '參考文獻', '参考资料', '參考資料', 
    '外部链接', '外部連結', '参见', '參見', 
    '注释', '註釋', '延伸阅读', '延伸閱讀', 
    'References', 'External links', 'See also', 
    'Notes', 'Bibliography', 'Further reading',
    'Source', '来源', '來源'
}

# Helper to generate stable IDs
def make_id(prefix, text):
    hash_suffix = hashlib.md5(text.encode('utf-8')).hexdigest()[:8].upper()
    return f"{prefix}_{hash_suffix}"

# Helper to safely get attributes from Entry object
def get_attr(obj, attr_name, method_name):
    if hasattr(obj, attr_name):
        return getattr(obj, attr_name)
    elif hasattr(obj, method_name):
        return getattr(obj, method_name)()
    return None


def process_chunk(chunk_args):
    """
    Worker function to process a range of ZIM entries.
    Args:
        chunk_args: tuple (start_index, count, zim_path, worker_idx)
    Returns:
        tuple (spaces_count, kus_count) - Metrics only, data is written to disk
    """
    start_index, count, zim_path, worker_idx = chunk_args
    
    # Define temporary output files (JSON Lines format for easier merging)
    temp_dir = os.path.join(OUTPUT_DIR, "temp_workers")
    spaces_file = os.path.join(temp_dir, f"spaces_{worker_idx}.jsonl")
    kus_file = os.path.join(temp_dir, f"kus_{worker_idx}.jsonl")
    
    # Initialize local ZIM instance (libzim objects often not picklable)
    zim_service = ZimService.initialize(zim_path)
    archive = zim_service._archive
    
    end_index = start_index + count
    # Determine safe upper bound
    if hasattr(archive, 'get_entry_count'):
        total = archive.get_entry_count()
    elif hasattr(archive, 'all_entry_count'):
        total = archive.all_entry_count
    else:
        total = 4000000 # fallback
        
    if end_index > total:
        end_index = total

    print(f"[Worker {worker_idx}] Processing range {start_index} to {end_index}...")
    
    processed_count = 0
    spaces_cnt = 0
    kus_cnt = 0
    
    # Open files for writing
    with open(spaces_file, 'w', encoding='utf-8') as f_spaces, \
         open(kus_file, 'w', encoding='utf-8') as f_kus:
         
        for i in range(start_index, end_index):
            try:
                processed_count += 1
                if processed_count % 500 == 0:
                     # print(f"[Worker {worker_idx}] Progress: {processed_count}/{count}")
                     pass

                entry = archive._get_entry_by_id(i)
                
                # Extract path/url
                path = get_attr(entry, 'url', 'get_path')
                if not path: 
                    if hasattr(entry, 'path'): path = entry.path
                    else: path = "unknown"

                # Filter: Only standard articles (Namespace 'A/')
                if not path.startswith('A/'):
                    continue
                    
                # Extract title
                title = get_attr(entry, 'title', 'get_title')
                if not title: 
                    title = path.replace('A/', '').replace('_', ' ')
                
                # --- LEVEL 1: Article as ZONE ---
                zone_id = make_id("ZONE", title)
                
                zone_obj = {
                    "space_id": zone_id,
                    "name": title,
                    "parent_id": "SCENIC_WIKI",   
                    "scenic_id": "SCENIC_WIKI",
                    "type": "zone",             
                    "description": "维基百科条目",
                    "metadata": {"wiki_path": path, "source": "zim"}
                }
                f_spaces.write(json.dumps(zone_obj, ensure_ascii=False) + "\n")
                spaces_cnt += 1
                
                # --- Parse Content Structure ---
                item = entry.get_item()
                # Note: Parsing implies CPU work
                soup = BeautifulSoup(bytes(item.content), 'html.parser')
                
                current_space_id = zone_id
                current_h2_id = zone_id
                content_buffer = []
                current_title_context = f"{title}"
                
                # Local helper for flushing
                def parse_flush(f_handle, space_id, buffer, context_title):
                    if not buffer: return 0
                    text_content = "\n".join(buffer).strip()
                    if len(text_content) < 5: return 0
                    
                    ku_id = make_id("KU", space_id + text_content[:20])
                    ku_obj = {
                        "ku_id": ku_id,
                        "scenic_id": "SCENIC_WIKI",
                        "space_id": space_id,
                        "topic": "百科",
                        "text": f"【{context_title}】\n{text_content}",
                        "assets": [],
                        "keywords": [title],
                        "metadata": {"source": "zim", "length": len(text_content)}
                    }
                    f_handle.write(json.dumps(ku_obj, ensure_ascii=False) + "\n")
                    return 1

                elements = soup.find_all(['h2', 'h3', 'p', 'ul', 'ol', 'table'])
                skipping_section = False
                
                for el in elements:
                    tag = el.name
                    text = el.get_text().strip()
                    
                    if tag in ['h2', 'h3']:
                        if not skipping_section:
                            kus_cnt += parse_flush(f_kus, current_space_id, content_buffer, current_title_context)
                            content_buffer = []

                        section_name = text.replace('[编辑]', '').replace('[編輯]', '').strip()
                        if not section_name: section_name = "未命名章节"
                        
                        is_ignored = section_name in IGNORED_HEADERS
                        if not is_ignored:
                            if len(section_name) < 10 and any(h in section_name for h in IGNORED_HEADERS):
                                is_ignored = True
                        
                        if is_ignored:
                            skipping_section = True
                            continue
                        else:
                            skipping_section = False

                        spot_id = make_id(f"SPOT_{zone_id}", section_name)
                        
                        parent_id = zone_id
                        if tag == 'h2':
                            current_h2_id = spot_id
                            parent_id = zone_id
                        elif tag == 'h3':
                            parent_id = current_h2_id
                        
                        spot_obj = {
                            "space_id": spot_id,
                            "name": section_name,
                            "parent_id": parent_id,
                            "scenic_id": "SCENIC_WIKI",
                            "type": "spot",
                            "description": section_name,
                            "metadata": {"tag": tag}
                        }
                        f_spaces.write(json.dumps(spot_obj, ensure_ascii=False) + "\n")
                        spaces_cnt += 1
                        
                        current_space_id = spot_id
                        current_title_context = f"{title} - {section_name}"

                    else:
                        if not skipping_section and text:
                            content_buffer.append(text)

                if not skipping_section:
                     kus_cnt += parse_flush(f_kus, current_space_id, content_buffer, current_title_context)
                     
            except Exception as e:
                # print(f"Error in {i}: {e}")
                continue
            
    return spaces_cnt, kus_cnt

def merge_files(output_file, temp_dir, prefix, start_char, end_char):
    """Merge JSONL files into a single JSON Array file without loading all to RAM"""
    print(f"Merging {prefix} files into {output_file}...")
    
    files = sorted([f for f in os.listdir(temp_dir) if f.startswith(prefix) and f.endswith('.jsonl')])
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write(start_char + "\n")
        first_item = True
        
        for fname in files:
            file_path = os.path.join(temp_dir, fname)
            with open(file_path, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line: continue
                    
                    if not first_item:
                        f_out.write(",\n")
                    f_out.write("    " + line)
                    first_item = False
            
            # Optional: Remove temp file after merge to free space
            # os.remove(file_path)
            
        f_out.write("\n" + end_char)
    print(f"✅ Generated {output_file} (Merged from {len(files)} parts)")

def run_multiprocess_extraction():
    # Setup temp directory
    temp_dir = os.path.join(OUTPUT_DIR, "temp_workers")
    if os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    # 1. scenics.json - 定义虚拟大景区
    scenics_data = [
        {
            "scenic_id": "SCENIC_WIKI",
            "name": "全息百科知识库",
            "description": "存储通用知识的虚拟空间，包含历史、科技、文化等多个主题展区。",
            "location": {"lat": 0.0, "lng": 0.0}, 
            "city_code": "VIRTUAL",
            "tags": ["百科", "知识库", "虚拟"]
        }
    ]
    path = os.path.join(OUTPUT_DIR, "scenics.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(scenics_data, f, indent=4, ensure_ascii=False)
    print(f"✅ Generated {path}")

    # Setup Multiprocessing
    # Limit number of articles (-1 for all)
    MAX_ARTICLES = -1 # Process ALL articles
    START_INDEX = 0

    
    # Get total count first
    zim_service = ZimService.initialize(ZIM_PATH)
    archive = zim_service._archive
    if hasattr(archive, 'all_entry_count'):
        total_available = archive.all_entry_count
    else:
        total_available = 4000000
    
    final_count = total_available
    if MAX_ARTICLES > 0 and MAX_ARTICLES < total_available:
        final_count = MAX_ARTICLES

    # Configuration for parallelism
    # 🎯 STABILITY OPTIMIZATION: 
    # Do NOT use all cores. Parsing HTML is memory intensive.
    # IO accessing the single ZIM file is also a bottleneck.
    # 16-32 workers is typically the sweet spot for throughput vs stability.
    user_cores = multiprocessing.cpu_count()
    SAFE_WORKERS = 16 
    NUM_WORKERS = min(user_cores, SAFE_WORKERS)
    
    print(f"🔧 Configuration: {NUM_WORKERS} Workers (capped for stability).")

    chunk_size = (final_count - START_INDEX) // NUM_WORKERS
    if chunk_size < 1: chunk_size = 1
    
    chunks = []
    current_start = START_INDEX
    for i in range(NUM_WORKERS):
        chunks.append((current_start, chunk_size, ZIM_PATH, i))
        current_start += chunk_size
    
    # Handle remainder
    if current_start < (START_INDEX + final_count):
        last_chunk = chunks[-1]
        chunks[-1] = (last_chunk[0], last_chunk[1] + ((START_INDEX + final_count) - current_start), last_chunk[2], last_chunk[3])

    print(f"🚀 Starting Stable ETL with {NUM_WORKERS} workers for {final_count} items...")
    
    # Try to import tqdm for progress bar
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        print("Tip: Install 'tqdm' for a nice progress bar: pip install tqdm")

    total_spaces = 0
    total_kus = 0

    try:
        with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
            # Use imap_unordered to monitor progress as chunks finish
            # This is more responsive than map() which waits for everything
            iterator = pool.imap_unordered(process_chunk, chunks)
            
            if use_tqdm:
                # Progress bar acts on Chunks completed, not items (since chunks are large tasks)
                with tqdm(total=len(chunks), unit="chunk", desc="ETL Progress") as pbar:
                    for metrics in iterator:
                        total_spaces += metrics[0]
                        total_kus += metrics[1]
                        pbar.update(1)
            else:
                completed = 0
                for metrics in iterator:
                    completed += 1
                    total_spaces += metrics[0]
                    total_kus += metrics[1]
                    print(f"✅ Chunk {completed}/{len(chunks)} finished. (Extracted: {metrics[0]} zones, {metrics[1]} units)")
        
        print("\nWorkers finished. Merging output files...")
        
        # Merge files
        merge_files(os.path.join(OUTPUT_DIR, "spaces.json"), temp_dir, "spaces", "[", "]")
        merge_files(os.path.join(OUTPUT_DIR, "knowledge_units_sample.json"), temp_dir, "kus", "[", "]")
        
        print(f"🎉 ETL Complete! Total: {total_spaces} Spaces, {total_kus} KnowledgeUnits.")
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
        print("🧹 Cleaned up temporary files.")

    except KeyboardInterrupt:
        print("\n🛑 Extraction interrupted by user. Cleaning up pool...")
        # Pool creates daemon processes, they will be killed when main process exits
        # But we explicitly terminate for safety
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    run_multiprocess_extraction()
