from src.rag.service.graph_discovery_service import (
    search_published_entities,
    get_published_node_detail,
    get_published_neighborhood,
)


def search_entity(
    domain_id: str,
    keyword: str
):
    """
    搜索知识图谱实体
    """

    return search_published_entities(
        domain_id=domain_id,
        q=keyword,
        limit=10,
        offset=0,
    )



def get_entity_detail(
    domain_id: str,
    node_id: str
):
    """
    获取实体详情，包括事实和关系
    """

    return get_published_node_detail(
        domain_id,
        node_id,
        relation_limit=50
    )



def explore_entity(
    domain_id: str,
    node_id: str
):
    """
    查询实体关联知识
    """

    return get_published_neighborhood(
        domain_id,
        node_id,
        depth=2,
        limit=50
    )