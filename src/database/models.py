from sqlalchemy import Column, String, Text, Integer, Numeric, DateTime, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import TIMESTAMP, JSONB, UUID
from sqlalchemy.sql import func
from src.database.base import Base


class MedicalProfile(Base):
    __tablename__ = "medical_profiles"
    
    profile_id = Column(String(255), primary_key=True, comment="uuid")
    user_id = Column(String(255), ForeignKey("users.user_id"), nullable=False, comment="外键 用户uuid")
    allergen_flags = Column(Text, nullable=False, comment="存储过敏原数组（如 ['peanuts','shellfish','gluten']）")
    dietary_restrictions = Column(Text, comment="存储非医疗性的饮食偏好数组（如 ['vegan','halal']）")
    mobility_restrictions = Column(Integer, comment="0/1（1标志行动不便）")
    create_time = Column(TIMESTAMP(6), comment="创建时间")
    update_time = Column(TIMESTAMP(6), comment="更新时间")


class MerchantProduct(Base):
    __tablename__ = "merchant_product"
    
    product_id = Column(String(255), primary_key=True, comment="uuid")
    merchant_id = Column(String(255), ForeignKey("merchants.merchant_id"), comment="商户uuid")
    product_name = Column(String(255), comment="商品名称")
    category = Column(String(255), comment="商品类别，“食品/纪念品”")
    desc = Column(Text, comment="商品描述，用于生成剧本中的道具描述")
    price = Column(Numeric(10, 2), comment="真实价格")
    image_url = Column(String(255), comment="商品图像")
    create_time = Column(TIMESTAMP(6), comment="创建时间")
    update_time = Column(TIMESTAMP(6), comment="更新时间")
    stock = Column(Integer, default=0, nullable=False, comment="库存")
    sold = Column(Integer, default=0, nullable=False, comment="核销")
    ingredients = Column(JSONB, comment="食品成分，存储为JSON格式")


class GameTeam(Base):
    __tablename__ = "game_team"
    
    team_id = Column(String(255), primary_key=True, comment="uuid")
    binding_code = Column(String(255), unique=True, nullable=False, comment="易于输入的短码（如“ZB888”） 生成时需保证唯一")
    leader_id = Column(String(255), comment="团队主导者id，通常是导游")
    size = Column(Integer, comment="团队规模")
    current_status = Column(Integer, comment="团队游玩状态 外键")
    aggregated_allergens = Column(Text, comment="当游客加入team时，业务层处理合并并更新到数据库，并存放到redis中，这里仅作为备份")
    create_time = Column(TIMESTAMP(6), comment="创建时间")
    update_time = Column(TIMESTAMP(6), comment="更新时间")


class UserMerchant(Base):
    __tablename__ = "user_merchant"
    
    user_id = Column(String(255), primary_key=True, comment="uuid")
    merchant_name = Column(String(255), comment="商户名称")
    location = Column(String(255), comment="商户位置")
    merchant_desc = Column(Text, comment="记录商户的类别和主要销售服务")
    merchant_banner_pic = Column(String(255), comment="记录商户的门头图像")
    business_hours = Column(String(255), comment="营业时间（格式 hh:mm:ss-hh:mm:ss）业务层保证格式校验")
    api_key = Column(String(255), comment="商户端App的认证密钥")
    create_time = Column(TIMESTAMP(6), comment="创建时间")
    update_time = Column(TIMESTAMP(6), comment="更新时间")


class ScriptTemplate(Base):
    __tablename__ = "script_templates"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    name = Column(String(100), nullable=False)
    style = Column(String(50), nullable=False)
    suitable_people = Column(Integer, nullable=False)
    template = Column(JSONB, nullable=False)
    created_by = Column(String(50), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())
    updated_at = Column(TIMESTAMP, nullable=False, server_default=func.now(), onupdate=func.now())


class GeneratedScript(Base):
    __tablename__ = "generated_scripts"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    team_id = Column(UUID(as_uuid=True), nullable=False)
    template_id = Column(UUID(as_uuid=True))
    script = Column(JSONB, nullable=False)
    status = Column(String(30), nullable=False, default='generated')
    created_at = Column(TIMESTAMP, nullable=False, server_default=func.now())

