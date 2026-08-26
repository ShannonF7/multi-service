'''入库质检：
    receive_payload
    ↓
    upsert_base_data
    ↓
    generate_chunks
    ↓
    rough_filter
    ↓
    llm_extract_facts
    ↓
    conflict_detect
    ↓
    wait_human_review
    ↓
    build_embeddings


'''