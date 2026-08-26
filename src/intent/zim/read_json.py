import ijson

path = "/home/zhangbi/Zhangbi_Traveler/DataBase/Search_Update_Context/json/pgvector_optimized/mock_scenic_data/spaces.json"

# with open(path, "rb") as f:
#     for _ in ijson.items(f, "item"):
#         count += 1
with open(path, "rb") as f:
    item = next(ijson.items(f, "item"))
    print(item["id"])
    print(item.get("space_id"))
    print(item.get("zone_id"))
    print(item.get("spot_id"))