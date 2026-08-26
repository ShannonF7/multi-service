# Constants for RAG module
# 放常量

INTENT_TYPES = [
    "qa",
    "spatial",
    "visual_similar",
    "text_to_image",
    "service",
    "hybrid",
    "chat",
]

RELATION_LAYER_SPATIAL = "spatial"
RELATION_LAYER_SEMANTIC = "semantic"

INTENT_WEIGHTS = {
    "qa": 1.0,
    "spatial": 0.8,
    "visual_similar": 0.7,
    "text_to_image": 0.6,
    "service": 0.5,
    "hybrid": 0.9,
    "chat": 0.4,
}
