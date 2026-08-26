from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from urllib.parse import quote
from src.core.config import settings
import logging
import time

logger = logging.getLogger(__name__)

# 为 SSH 隧道保留一个全局引用，防止其被垃圾回收
tunnel = None

# 引擎和会话延迟初始化，避免在模块导入时因隧道未就绪而创建失败的 engine
engine = None
SessionLocal = None


# 动态生成数据库连接 URL，并在使用 SSH 隧道时提供重试
def construct_db_url(retry: int = 3, retry_interval: float = 1.0):
    global tunnel

    db_host = settings.DB_HOST
    db_port = settings.DB_PORT

    if settings.USE_SSH_TUNNEL:
        try:
            from sshtunnel import SSHTunnelForwarder
        except ImportError:
            logger.error("sshtunnel package not found. Please run 'pip install sshtunnel'.")
            raise

        # 尝试建立隧道（带重试），避免短暂网络抖动导致失败
        last_exc = None
        for attempt in range(1, retry + 1):
            try:
                logger.info(f"Starting SSH Tunnel to {settings.SSH_HOST}:{settings.SSH_PORT} (attempt {attempt})...")
                tunnel = SSHTunnelForwarder(
                    (settings.SSH_HOST, settings.SSH_PORT),
                    ssh_username=settings.SSH_USER,
                    ssh_password=settings.SSH_PASSWORD,
                    remote_bind_address=(settings.REMOTE_DB_HOST, settings.REMOTE_DB_PORT),
                    local_bind_address=("127.0.0.1", 0),  # 自动分配本地可用端口
                )
                tunnel.start()
                db_host = "127.0.0.1"
                db_port = tunnel.local_bind_port
                logger.info(f"SSH Tunnel established. Local port: {db_port}")
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                logger.error(f"Failed to start SSH Tunnel (attempt {attempt}): {e}")
                # 尝试清理上次的 tunnel 对象（如果有）
                try:
                    if tunnel is not None:
                        tunnel.close()
                except Exception:
                    pass
                tunnel = None
                if attempt < retry:
                    time.sleep(retry_interval)

        if last_exc is not None:
            # 无法建立隧道时抛出异常，让调用者能做出更明确的处理或重试策略
            raise RuntimeError(f"Could not establish SSH tunnel after {retry} attempts: {last_exc}")

    encoded_password = quote(settings.DB_PASSWORD)
    db_url = f"postgresql://{settings.DB_USER}:{encoded_password}@{db_host}:{db_port}/{settings.DB_NAME}"
    logger.debug(f"Constructed Database URL: postgresql://{settings.DB_USER}:<password>@{db_host}:{db_port}/{settings.DB_NAME}")
    return db_url


def init_engine():
    """延迟初始化 SQLAlchemy 引擎与 SessionLocal。"""
    global engine, SessionLocal
    if engine is not None:
        return

    db_url = construct_db_url()
    engine = create_engine(
        db_url,
        pool_size=10,
        max_overflow=20,
        pool_timeout=30,
        pool_recycle=1800,
        # 当连接池中的连接被远端关闭（例如隧道断开、网络短暂中断）时，
        # pool_pre_ping 会在取连接前执行简单的 ping（SELECT 1）来检测连接是否可用，
        # 如果不可用 SQLAlchemy 会自动重建连接，从而减少 OperationalError
        pool_pre_ping=True,
    )

    SessionLocal = scoped_session(
        sessionmaker(autocommit=False, autoflush=False, bind=engine)
    )


def get_db():
    """获取数据库会话的依赖项；首次调用时会初始化引擎（如果尚未初始化）。"""
    global SessionLocal
    if SessionLocal is None:
        init_engine()

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()